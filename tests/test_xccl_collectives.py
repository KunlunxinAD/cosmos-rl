# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch
import torch.multiprocessing as mp
from cosmos_rl.utils.pyxccl import (
    create_nccl_uid,
    create_nccl_comm,
    nccl_send,
    nccl_recv,
    nccl_broadcast,
    nccl_allreduce,
    nccl_reducescatter,
    nccl_allgather,
)
from cosmos_rl.utils.pyxccl_wrapper import xcclRedOpTypeEnum


def setup_xccl_comm(rank, world_size, xccl_uid):
    """Setup NCCL communicator for a process."""
    # Set device for this process (now using the visible device)
    torch.cuda.set_device(rank)  # Always use first visible device
    # Create XCCL communicator
    comm_idx = create_nccl_comm(xccl_uid, rank, world_size)
    return comm_idx


class TestXCCLBidirectionalSendRecv(unittest.TestCase):
    @staticmethod
    def run_bidirectional_sender(rank, world_size, xccl_uid, dtypes):
        """Run sender part of bidirectional XCCL send/recv test."""
        comm_idx = setup_xccl_comm(rank, world_size, xccl_uid)

        for dtype in dtypes:
            # Create test tensor
            tensor_size = 1000
            send_tensor = torch.ones(
                tensor_size, dtype=dtype, device=f"cuda:{rank}"
            ) * (rank + 1)
            recv_tensor = torch.zeros(tensor_size, dtype=dtype, device=f"cuda:{rank}")

            send_rank = 0
            # Send to other rank and receive from other rank
            other_rank = 1 - rank
            if rank == send_rank:
                nccl_send(send_tensor, other_rank, comm_idx)
                nccl_recv(recv_tensor, other_rank, comm_idx)
            else:
                nccl_recv(recv_tensor, other_rank, comm_idx)
                nccl_send(recv_tensor, other_rank, comm_idx)

            # Verify received data
            expected = torch.ones(tensor_size, dtype=dtype, device=f"cuda:{rank}") * (
                send_rank + 1
            )
            assert torch.allclose(recv_tensor, expected), f"SendRecv failed for dtype {dtype}"

    def test_xccl_bidirectional_send_recv(self):
        """Test bidirectional XCCL send/recv operations between two processes with different CUDA devices."""
        print("Testing XCCL Bidirectional Send/Recv...")

        world_size = 2

        # Create XCCL unique ID
        xccl_uid = create_nccl_uid()

        # Define functions for each process (same function but different rank)
        functions = self.run_bidirectional_sender
        # Spawn processes with different functions
        dtypes = [
            torch.float32,
            torch.float16,
            torch.int32,
            torch.int64,
            torch.uint8,
            # torch.int8,
            torch.bfloat16,
            torch.float64,
        ]
        mp.spawn(
            functions,
            args=(world_size, xccl_uid, dtypes),
            nprocs=world_size,
            join=True,
        )


class TestXCCLBroadcast(unittest.TestCase):
    @staticmethod
    def run_broadcast(rank, world_size, xccl_uid, dtypes):
        """Run broadcast test for different data types."""
        comm_idx = setup_xccl_comm(rank, world_size, xccl_uid)

        for dtype in dtypes:
            # Test broadcasting from each rank
            for root_rank in range(world_size):
                # Create test tensor
                tensor_size = 1000
                if rank == root_rank:  # Root rank
                    # Create tensor with unique values based on root rank
                    tensor = torch.arange(
                        tensor_size, dtype=dtype, device=f"cuda:{rank}"
                    ) * (root_rank + 1)
                else:
                    # Create empty tensor for receiving
                    tensor = torch.zeros(
                        tensor_size, dtype=dtype, device=f"cuda:{rank}"
                    )

                # Perform broadcast from current root rank
                nccl_broadcast(tensor, root_rank, comm_idx)

                # Verify received data
                expected = torch.arange(
                    tensor_size, dtype=dtype, device=f"cuda:{rank}"
                ) * (root_rank + 1)
                assert torch.allclose(
                    tensor, expected
                ), f"Broadcast from rank {root_rank} failed for dtype {dtype}"

    def test_xccl_broadcast(self):
        """Test XCCL broadcast operations between multiple processes with different CUDA devices."""
        print("Testing XCCL Broadcast...")

        world_size = 8

        # Create XCCL unique ID
        xccl_uid = create_nccl_uid()

        # Define data types to test
        dtypes = [
            torch.float32,
            torch.float16,
            torch.int32,
            torch.int64,
            torch.uint8,
            # torch.int8,
            torch.bfloat16,
            torch.float64,
        ]

        # Spawn processes
        mp.spawn(
            self.run_broadcast,
            args=(world_size, xccl_uid, dtypes),
            nprocs=world_size,
            join=True,
        )


class TestXCCLAllreduce(unittest.TestCase):
    @staticmethod
    def run_allreduce(rank, world_size, xccl_uid, dtypes):
        """Run allreduce test for different data types."""
        comm_idx = setup_xccl_comm(rank, world_size, xccl_uid)

        for dtype in dtypes:
            # Create test tensor
            tensor_size = 1000
            # Each rank creates a tensor with its rank value
            tensor = torch.ones(tensor_size, dtype=dtype, device=f"cuda:{rank}") * (
                rank + 1
            )
            op = xcclRedOpTypeEnum.from_torch(torch.distributed.ReduceOp.SUM)
            # Perform allreduce (sum)
            nccl_allreduce(tensor, tensor, op, comm_idx)

            # Verify result
            # For 4 ranks, the sum should be 1 + 2 + 3 + 4 = 10
            # For n rank, the sum should be (n + 1) * n / 2
            expected_sum = (
                torch.ones(tensor_size, dtype=dtype, device=f"cuda:{rank}") * int((world_size + 1) * world_size / 2)
            )
            assert torch.allclose(tensor, expected_sum), f"AllReduce failed for dtype {dtype}"

    def test_xccl_allreduce(self):
        """Test XCCL allreduce operations between multiple processes with different CUDA devices."""
        print("Testing XCCL AllReduce...")

        world_size = 8

        # Create NCCL unique ID
        xccl_uid = create_nccl_uid()

        # Define data types to test
        dtypes = [
            torch.float32,
            torch.float16,
            torch.int32,
            torch.int64,
            torch.uint8,
            # torch.int8,
            torch.bfloat16,
            torch.float64,
        ]

        # Spawn processes
        mp.spawn(
            self.run_allreduce,
            args=(world_size, xccl_uid, dtypes),
            nprocs=world_size,
            join=True,
        )


class TestXCCLReduceScatter(unittest.TestCase):
    @staticmethod
    def run_reducescatter(rank, world_size, xccl_uid, dtypes):
        comm_idx = setup_xccl_comm(rank, world_size, xccl_uid)
        for dtype in dtypes:
            tensor_size = 1000 * world_size
            send_tensor = torch.ones(tensor_size, dtype=dtype, device=f"cuda:{rank}") * (rank + 1)
            recv_tensor = torch.zeros(1000, dtype=dtype, device=f"cuda:{rank}")
            op = xcclRedOpTypeEnum.from_torch(torch.distributed.ReduceOp.SUM)
            nccl_reducescatter(send_tensor, recv_tensor, op, comm_idx)
            expected = torch.ones(1000, dtype=dtype, device=f"cuda:{rank}") * int((world_size + 1) * world_size / 2)
            assert torch.allclose(recv_tensor, expected), f"ReduceScatter failed for dtype {dtype}"

    def test_xccl_reducescatter(self):
        print("Testing XCCL ReduceScatter...")

        world_size = 8

        xccl_uid = create_nccl_uid()

        dtypes = [
            torch.float32,
            torch.float16,
            torch.int32,
            torch.int64,
            torch.uint8,
            # torch.int8,
            torch.bfloat16,
            torch.float64,
        ]
        
        mp.spawn(
            self.run_reducescatter,
            args=(world_size, xccl_uid, dtypes),
            nprocs=world_size,
            join=True,
        )

class TestXCCLAllGather(unittest.TestCase):
    @staticmethod
    def run_allgather(rank, world_size, xccl_uid, dtypes):
        comm_idx = setup_xccl_comm(rank, world_size, xccl_uid)
        for dtype in dtypes:
            tensor_size = 1000
            send_tensor = torch.ones(tensor_size, dtype=dtype, device=f"cuda:{rank}") * (rank + 1)
            recv_tensor = torch.zeros(tensor_size * world_size, dtype=dtype, device=f"cuda:{rank}")
            nccl_allgather(send_tensor, recv_tensor, comm_idx)
            expected = torch.cat([
                torch.ones(tensor_size, dtype=dtype, device=f"cuda:{rank}") * (r + 1)
                for r in range(world_size)
            ])
            assert torch.allclose(recv_tensor, expected), f"AllGather failed for dtype {dtype}"

    def test_xccl_allgather(self):
        print("Testing XCCL AllGather...")

        world_size = 8

        xccl_uid = create_nccl_uid()

        dtypes = [
            torch.float32,
            torch.float16,
            torch.int32,
            torch.int64,
            torch.uint8,
            # torch.int8,
            torch.bfloat16,
            torch.float64,
        ]

        mp.spawn(
            self.run_allgather,
            args=(world_size, xccl_uid, dtypes),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
