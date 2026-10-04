# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import deque
from unittest import mock

import msgpack
import pytest

from megatron.core.inference.headers import Headers
from tests.unit_tests.inference.coordinator_test_utils import make_coordinator_direct


def _message(sender, header, *payload):
    return [sender, msgpack.packb([header.value, *payload], use_bin_type=True)]


def _coordinator(messages):
    coordinator = make_coordinator_direct(data_parallel_size=2)
    coordinator.state = coordinator.CoordinatorState.RUNNING
    coordinator.next_request_id = 0
    coordinator.schedule_records = None
    coordinator.detokenize = mock.Mock()
    coordinator.router_socket = mock.Mock()
    queue = deque(messages)
    coordinator.router_socket.recv_multipart.side_effect = queue.popleft
    coordinator._send_to_engine = mock.Mock(return_value=True)
    return coordinator


def _replies(coordinator, header):
    replies = []
    for call in coordinator.router_socket.send_multipart.call_args_list:
        identity, packed = call.args[0]
        payload = msgpack.unpackb(packed, raw=False)
        if payload[0] == header.value:
            replies.append((identity, payload))
    return replies


def test_partial_reply_retains_routing_and_load_until_final():
    coordinator = _coordinator(
        [
            _message(b"client", Headers.CONNECT),
            _message(b"client", Headers.SUBMIT_REQUEST, 7, [1, 2], {}),
            _message(
                b"rank_0",
                Headers.ENGINE_REPLY_PARTIAL,
                [{"request_id": 0, "new_tokens": [11], "new_log_probs": [-0.5]}],
            ),
            _message(b"rank_0", Headers.ENGINE_REPLY, [{"request_id": 0}]),
            _message(b"client", Headers.SHUTDOWN),
        ]
    )
    observed_load = []

    def observe_reply(frames):
        payload = msgpack.unpackb(frames[1], raw=False)
        if payload[0] == Headers.ENGINE_REPLY_PARTIAL.value:
            observed_load.append(coordinator._pending_counts.tolist())
            assert coordinator.client_request_to_request_id == {(b"client", 7): 0}
            assert coordinator.request_id_to_rank == {0: b"rank_0"}

    coordinator.router_socket.send_multipart.side_effect = observe_reply
    coordinator.start()

    assert observed_load == [[1, 0]]
    assert _replies(coordinator, Headers.ENGINE_REPLY_PARTIAL) == [
        (
            b"client",
            [
                Headers.ENGINE_REPLY_PARTIAL.value,
                7,
                {"request_id": 0, "new_tokens": [11], "new_log_probs": [-0.5]},
            ],
        )
    ]
    coordinator.detokenize.assert_called_once_with({"request_id": 0})
    assert coordinator._pending_counts.tolist() == [0, 0]
    assert not coordinator.client_request_to_request_id
    assert not coordinator.request_id_to_client_id
    assert not coordinator.request_id_to_client_request_id
    assert not coordinator.request_id_to_rank


def test_abort_uses_client_identity_and_does_not_release_pending_state():
    coordinator = _coordinator(
        [
            _message(b"client-A", Headers.CONNECT),
            _message(b"client-B", Headers.CONNECT),
            _message(b"client-A", Headers.SUBMIT_REQUEST, 7, [1], {}),
            _message(b"client-B", Headers.SUBMIT_REQUEST, 7, [2], {}),
            _message(b"unknown", Headers.ABORT_REQUEST, 7),
            _message(b"client-A", Headers.ABORT_REQUEST, 999),
            _message(b"client-B", Headers.ABORT_REQUEST, 7),
            _message(b"client-A", Headers.SHUTDOWN),
        ]
    )
    coordinator.start()

    assert coordinator._send_to_engine.call_count == 3
    assigned_rank, packed = coordinator._send_to_engine.call_args.args
    assert assigned_rank == b"rank_1"
    assert msgpack.unpackb(packed, raw=False) == [Headers.ABORT_REQUEST.value, 1]
    assert coordinator.client_request_to_request_id == {(b"client-A", 7): 0, (b"client-B", 7): 1}
    assert coordinator._pending_counts.tolist() == [1, 1]


def test_failed_routing_cleans_client_correlation():
    coordinator = _coordinator(
        [
            _message(b"client", Headers.CONNECT),
            _message(b"client", Headers.SUBMIT_REQUEST, 7, [1], {}),
        ]
    )
    coordinator._send_to_engine.return_value = False
    coordinator.start()

    assert coordinator._send_to_engine.call_count == 2
    assert not coordinator.client_request_to_request_id
    assert not coordinator.request_id_to_client_id
    assert not coordinator.request_id_to_client_request_id
    assert coordinator._pending_counts.tolist() == [0, 0]


def test_removed_engine_partial_is_ignored_but_unknown_engine_is_rejected():
    coordinator = _coordinator(
        [
            _message(b"client", Headers.CONNECT),
            _message(b"removed", Headers.ENGINE_REPLY_PARTIAL, []),
            _message(b"client", Headers.SHUTDOWN),
        ]
    )
    coordinator.removed_engine_identities.add(b"removed")
    coordinator.start()
    assert not _replies(coordinator, Headers.ENGINE_REPLY_PARTIAL)

    coordinator = _coordinator([_message(b"unknown", Headers.ENGINE_REPLY_PARTIAL, [])])
    with pytest.raises(AssertionError, match="never-connected sender"):
        coordinator.start()


def test_streaming_headers_preserve_existing_wire_values():
    existing = [
        Headers.CONNECT,
        Headers.CONNECT_ACK,
        Headers.SUBMIT_REQUEST,
        Headers.ENGINE_REPLY,
        Headers.PAUSE,
        Headers.UNPAUSE,
        Headers.SUSPEND,
        Headers.RESUME,
        Headers.SET_GENERATION_EPOCH,
        Headers.STOP,
        Headers.DISCONNECT,
        Headers.SHUTDOWN,
        Headers.TP_BROADCAST,
    ]
    assert [header.value for header in existing] == list(range(1, 14))
    assert Headers.ENGINE_REPLY_PARTIAL.value == 14
    assert Headers.ABORT_REQUEST.value == 15
