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

import ctypes
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed import ReduceOp

from cosmos_rl.utils.logging import logger


# === export types and functions from nccl to Python ===
# for the original nccl definition, please check
# https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in

xcclResult_t = ctypes.c_int
# xcclComm_t = ctypes.c_void_p
xcclContext_t = ctypes.c_void_p 


class xcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


xpuStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

xcclDataType_t = ctypes.c_int

# Do not support in XCCL:
# --- NCCL config struct (complete v22700) ---
# Used in ncclCommInitRankConfig 
# class ncclConfig_t(ctypes.Structure):
#     _fields_ = [
#         ("size", ctypes.c_size_t),  # sizeof(ncclConfig_t)
#         ("magic", ctypes.c_uint),  # constant magic
#         ("version", ctypes.c_uint),  # NCCL version code, e.g. 22703
#         ("blocking", ctypes.c_int),  # whether operations are blocking (0 / 1)
#         ("cgaClusterSize", ctypes.c_int),
#         ("minCTAs", ctypes.c_int),
#         ("maxCTAs", ctypes.c_int),
#         ("netName", ctypes.c_char_p),
#         ("splitShare", ctypes.c_int),
#         ("trafficClass", ctypes.c_int),
#         ("commName", ctypes.c_char_p),
#         ("collnetEnable", ctypes.c_int),
#         ("CTAPolicy", ctypes.c_int),
#         ("shrinkShare", ctypes.c_int),
#         ("nvlsCTAs", ctypes.c_int),
#     ]


class xcclDataTypeEnum:
    # Supported types in XCCL:
    xcclUint8 = 6 # ncclUint8 = 1
    xcclInt32 = 4 # ncclInt32 = 2
    xcclInt = 4 # ncclInt = 2
    xcclInt64 = 5 # ncclInt64 = 4
    xcclFloat16 = 1 # ncclFloat16 = 6
    xcclHalf = 1 # ncclHalf = 6
    xcclFloat32 = 0 # ncclFloat32 = 7
    xcclFloat = 0 # ncclFloat = 7
    xcclFloat64 = 3 # ncclFloat64 = 8
    xcclBfloat16 = 2 # ncclBfloat16 = 9
    # Not supported types in XCCL:
    # ncclInt8 = 0
    # ncclChar = 0
    # ncclUint32 = 3
    # ncclUint64 = 5
    # ncclDouble = 8
    # ncclFloat8e4m3 = 10
    # ncclFloat8e5m2 = 11

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            raise TypeError("XCCL does not support int8 data type.")
            # return cls.xcclInt8
        if dtype == torch.uint8:
            return cls.xcclUint8
        if dtype == torch.int32:
            return cls.xcclInt32
        if dtype == torch.int64:
            return cls.xcclInt64
        if dtype == torch.float16:
            return cls.xcclFloat16
        if dtype == torch.float32:
            return cls.xcclFloat32
        if dtype == torch.float64:
            return cls.xcclFloat64
        if dtype == torch.bfloat16:
            return cls.xcclBfloat16
        if dtype == torch.float8_e4m3fn:
            raise TypeError("XCCL does not support float8_e4m3fn data type.")
            # return cls.xcclFloat8e4m3
        if dtype == torch.float8_e5m2:
            raise TypeError("XCCL does not support float8_e5m2 data type.")
            # return cls.xcclFloat8e5m2
        raise ValueError(f"Unsupported dtype: {dtype}")


xcclRedOp_t = ctypes.c_int


class xcclRedOpTypeEnum:
    xcclSum = 0
    xcclProd = 1
    xcclMin = 2
    xcclMax = 3
    # Note: xccl 没有 avg 操作
    # xcclAvg = 4
    xcclNumOps = 4
    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.xcclSum
        if op == ReduceOp.PRODUCT:
            return cls.xcclProd
        if op == ReduceOp.MAX:
            return cls.xcclMax
        if op == ReduceOp.MIN:
            return cls.xcclMin
        if op == ReduceOp.AVG:
            raise TypeError("XCCL does not support AVG reduction operation.")
            # return cls.xcclAvg
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: list[Any]


class XCCLLibrary:
    # names of optional functions (absence tolerated)
    # optional_functions = {"ncclCommInitRankConfig"}
    optional_functions = {
        # "ncclCommInitRankConfig", # falling back to xccl_init_rank
        # "ncclGetErrorString", # does not provide
        # "ncclCommGetAsyncError", # does not provide
    }
    exported_functions = [
        
        # Note: XCCL does not provide xcclGetErrorString
        # NV: const char* ncclGetErrorString(ncclResult_t result)
        # Function("ncclGetErrorString", ctypes.c_char_p, [xcclResult_t]),

        # Note: get_version functions are different in NCCL and XCCL, xccl_version_info returns void and prints to stdout
        # NV: ncclResult_t  ncclGetVersion(int *version);
        # XPU: void xccl_version_info();
        Function("_Z17xccl_version_infov", None, []),

        # NV: ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
        # XPU: XCCLResult_t xccl_get_unique_id(XCCLUniqueId* id);
        Function("_Z18xccl_get_unique_idP12XCCLUniqueId", xcclResult_t, [ctypes.POINTER(xcclUniqueId)]),

        # Note: XCCL does not provide xcclCommInitRankConfig; calls will fall back to xccl_init_rank.
        # Note: the parameter "UniqueID" is a pointer in XCCL, instead of a value type in NCCL
        # NV: ncclResult_t  ncclCommInitRank(
        # NV:   ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
        # XPU: XCCLResult_t xccl_init_rank(XCCLContext_t* ctx, int rank, int nranks, const XCCLUniqueId* id);
        # note that ncclComm_t is a pointer type, so the first argument
        # is a pointer to a pointer
        Function(
            "_Z14xccl_init_rankPPviiPK12XCCLUniqueId",
            xcclResult_t,
            [ctypes.POINTER(xcclContext_t), ctypes.c_int, ctypes.c_int, ctypes.POINTER(xcclUniqueId)],
        ),

        # NV: ncclResult_t  ncclAllReduce(
        # NV:   const void* sendbuff, void* recvbuff, size_t count,
        # NV:   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
        # NV:   cudaStream_t stream);
        # XPU: XCCLResult_t xccl_all_reduce(const XCCLContext_t ctx, const void* sendbuf, void* recvbuf,
        # XPU:   size_t count, XCCLDataType datatype, XCCLOp op, XPUStream stream);
        Function(
            "_Z15xccl_all_reducePvPKvS_m12XCCLDataType6XCCLOpS_",
            xcclResult_t,
            [
                xcclContext_t,
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                xcclDataType_t,
                xcclRedOp_t,
                xpuStream_t,
            ],
        ),

        # NV: ncclResult_t  ncclAllGather(
        # NV:   const void* sendbuff, void* recvbuff, size_t count,
        # NV:   ncclDataType_t datatype, ncclComm_t comm,
        # NV:   cudaStream_t stream);
        # XPU: XCCLResult_t xccl_all_gather(const XCCLContext_t ctx, const void* sendbuf, size_t sendcnt,
        # XPU:   void* recvbuf, XCCLDataType datatype, XPUStream stream);
        Function(
            "_Z15xccl_all_gatherPvPKvmS_12XCCLDataTypeS_",
            xcclResult_t,
            [
                xcclContext_t,
                buffer_type,
                ctypes.c_size_t,
                buffer_type,
                xcclDataType_t,
                xpuStream_t,
            ],
        ),

        # NV: ncclResult_t  ncclReduceScatter(
        # NV:   const void* sendbuff, void* recvbuff, size_t count,
        # NV:   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
        # NV:   cudaStream_t stream);
        # XPU: XCCLResult_t xccl_reduce_scatter(const XCCLContext_t ctx, const void* sendbuf, void* recvbuf,
        # XPU:   size_t recvcnt, XCCLDataType datatype, XCCLOp op, XPUStream stream);
        Function(
            "_Z19xccl_reduce_scatterPvPKvS_m12XCCLDataType6XCCLOpS_",
            xcclResult_t,
            [
                xcclContext_t,
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                xcclDataType_t,
                xcclRedOp_t,
                xpuStream_t,
            ],
        ),

        # NV: ncclResult_t  ncclSend(
        # NV:   const void* sendbuff, size_t count, ncclDataType_t datatype,
        # NV:   int dest, ncclComm_t comm, cudaStream_t stream);
        # XPU: XCCLResult_t xccl_send(const XCCLContext_t ctx, const void* sendbuf, size_t cnt,
        # XPU:   int peer, XCCLDataType datatype, XPUStream stream);
        Function(
            "_Z9xccl_sendPvPKvmi12XCCLDataTypeS_",
            xcclResult_t,
            [
                xcclContext_t,
                buffer_type,
                ctypes.c_size_t,
                ctypes.c_int,
                xcclDataType_t,
                xpuStream_t,
            ],
        ),

        # NV: ncclResult_t  ncclRecv(
        # NV:   void* recvbuff, size_t count, ncclDataType_t datatype,
        # NV:   int src, ncclComm_t comm, cudaStream_t stream);
        # XPU: XCCLResult_t xccl_recv(const XCCLContext_t ctx, void* recvbuf, size_t cnt,
        # XPU:   int peer, XCCLDataType datatype, XPUStream stream);
        Function(
            "_Z9xccl_recvPvS_mi12XCCLDataTypeS_",
            xcclResult_t,
            [
                xcclContext_t,
                buffer_type,
                ctypes.c_size_t,
                ctypes.c_int,
                xcclDataType_t,
                xpuStream_t,
            ],
        ),
        
        # NV: ncclResult_t ncclBroadcast(
        # NV:   const void* sendbuff, void* recvbuff, size_t count,
        # NV:   ncclDataType_t datatype, int root, ncclComm_t comm,
        # NV:   cudaStream_t stream);
        # XPU: XCCLResult_t xccl_broadcast(const XCCLContext_t ctx, const void* sendbuf, void* recvbuf,
        # XPU:   size_t count, XCCLDataType datatype, int root, XPUStream stream);
        Function(
            "_Z14xccl_broadcastPvPKvS_m12XCCLDataTypeiS_",
            xcclResult_t,
            [
                xcclContext_t,
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                xcclDataType_t,
                ctypes.c_int,
                xpuStream_t,
            ],
        ),
        
        # NV: ncclResult_t  ncclCommDestroy(ncclComm_t comm);
        # XPU: XCCLResult_t xccl_destroy_context(XCCLContext_t ctx);
        # be cautious! this is a collective call, it will block until all
        # processes in the communicator have called this function.
        # because Python object destruction can happen in random order,
        # it is better not to call it at all.
        Function("_Z20xccl_destroy_contextPv", xcclResult_t, [xcclContext_t]),

        # Note: XCCL does not have xcclCommGetAsyncError
        # NV: ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError);
        # NCCL async error query & abort (for enqueue-timeout protection)
        # Function(
        #     "ncclCommGetAsyncError",
        #     xcclResult_t,
        #     [xcclContext_t, ctypes.POINTER(xcclResult_t)],
        # ),

        # NV: ncclResult_t ncclCommAbort(ncclComm_t comm);
        # XPU: XCCLResult_t xccl_comm_abort(XCCLContext_t ctx);
        Function("_Z15xccl_comm_abortPv", xcclResult_t, [xcclContext_t]),

        # Note: XCCL does not provide ncclCommInitRankConfig, falling back to xccl_init_rank
        # NV: ncclResult_t  ncclCommInitRankConfig(
        # NV:   ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank,
        # NV:   ncclConfig_t* config);
        # Function(
        #     "ncclCommInitRankConfig",
        #     xcclResult_t,
        #     [
        #         ctypes.POINTER(xcclContext_t),
        #         ctypes.c_int,
        #         xcclUniqueId,
        #         ctypes.c_int,
        #         ctypes.POINTER(ncclConfig_t),
        #     ],
        # ),

        # NV: ncclResult_t ncclGroupStart()
        # XPU: XCCLResult_t xccl_group_start();
        Function(
            "_Z16xccl_group_startv",
            xcclResult_t,
            [],
        ),

        # NV: ncclResult_t ncclGroupEnd()
        # XPU: XCCLResult_t xccl_group_end();
        Function(
            "_Z14xccl_group_endv",
            xcclResult_t,
            [],
        ),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding functions
    path_to_funcs_mapping: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: str):
        try:
            if so_file not in XCCLLibrary.path_to_library_cache:
                lib = ctypes.CDLL(so_file)
                XCCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = XCCLLibrary.path_to_library_cache[so_file]
        except Exception:
            logger.error(f"Failed to load so file: {so_file} from XCCL library. ")
            raise

        if so_file not in XCCLLibrary.path_to_funcs_mapping:
            _funcs: dict[str, Any] = {}
            for func in XCCLLibrary.exported_functions:
                try:
                    f = getattr(self.lib, func.name)
                except AttributeError:
                    if func.name in XCCLLibrary.optional_functions:
                        logger.warning(
                            f"Optional XCCL symbol {func.name} not found; falling back to default behavior."
                        )
                        _funcs[func.name] = None
                        continue
                    raise
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            XCCLLibrary.path_to_funcs_mapping[so_file] = _funcs
        self._funcs = XCCLLibrary.path_to_funcs_mapping[so_file]

    def xcclGetErrorString(self, result: xcclResult_t) -> str:
        # Note: XCCL does not have xcclGetErrorString
        fn = self._funcs.get("ncclGetErrorString")
        if fn is None:
            # Return original error message
            logger.info(
                "XCCL does not have xcclGetErrorString function – returning original error info."
            )
            return result
        return fn(result).decode("utf-8")

    def XCCL_CHECK(self, result: xcclResult_t) -> None:
        """Raise RuntimeError if *result* is an error code."""
        if not xcclResultEnum.is_ok(int(result)):
            error_str = self.xcclGetErrorString(result)
            raise RuntimeError(f"XCCL error: {error_str}")
        
    def xcclGetVersion(self) -> str:
        # Note: xccl_version_info returns void and prints to stdout, cannot get version programmatically
        # Need to call xcclVersionInfo() separately to see version
        xcclVersionInfo()
        return "XCCL version info printed to stdout."

    def xcclVersionInfo(self):
        # Do not have return value
        self._funcs["_Z17xccl_version_infov"]()
        
    def xcclGetUniqueId(self) -> xcclUniqueId:
        unique_id = xcclUniqueId()
        self.XCCL_CHECK(self._funcs["_Z18xccl_get_unique_idP12XCCLUniqueId"](ctypes.byref(unique_id)))
        logger.debug(f"XCCL unique ID created. ID: {unique_id.internal}")
        return unique_id

    def xcclCommInitRank(
        self, world_size: int, unique_id: xcclUniqueId, rank: int
    ) -> xcclContext_t:
        comm = xcclContext_t()
        self.XCCL_CHECK(
            self._funcs["_Z14xccl_init_rankPPviiPK12XCCLUniqueId"](
                ctypes.byref(comm), rank, world_size, ctypes.byref(unique_id)
            )
        )
        return comm

    def xcclCommInitRankConfig(
        self,
        world_size: int,
        unique_id: xcclUniqueId,
        rank: int,
        blocking: int = 0,
    ) -> xcclContext_t:
        # Note: XCCL does not have xcclCommInitRankConfig, falls back to xccl_init_rank
        """Python wrapper for ncclCommInitRankConfig with *blocking* flag preset.
        
        Currently only exposes the *blocking* field (0 = non-blocking NCCL launch).
        Additional ncclConfig_t fields are kept at their default zeros for
        simplicity.
        """

        logger.info(
            "XCCL does not have xcclCommInitRankConfig function – falling back to xccl_init_rank. "
            "The 'blocking' parameter will be ignored."
        )
        return self.xcclCommInitRank(world_size, unique_id, rank)

    def xcclAllReduce(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: xcclContext_t,
        stream: xpuStream_t,
    ) -> None:
        # `datatype` actually should be `xcclDataType_t`
        # and `op` should be `xcclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.XCCL_CHECK(
            self._funcs["_Z15xccl_all_reducePvPKvS_m12XCCLDataType6XCCLOpS_"](
                comm, sendbuff, recvbuff, count, datatype, op, stream
            )
        )

    def xcclReduceScatter(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: xcclContext_t,
        stream: xpuStream_t,
    ) -> None:
        # `datatype` actually should be `xcclDataType_t`
        # and `op` should be `xcclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.XCCL_CHECK(
            self._funcs["_Z19xccl_reduce_scatterPvPKvS_m12XCCLDataType6XCCLOpS_"](
                comm, sendbuff, recvbuff, count, datatype, op, stream
            )
        )

    def xcclAllGather(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        comm: xcclContext_t,
        stream: xpuStream_t,
    ) -> None:
        # `datatype` actually should be `xcclDataType_t`
        # which is an aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.XCCL_CHECK(
            self._funcs["_Z15xccl_all_gatherPvPKvmS_12XCCLDataTypeS_"](
                comm, sendbuff, count, recvbuff, datatype, stream
            )
        )

    def xcclSend(
        self,
        sendbuff: buffer_type,
        count: int,
        datatype: int,
        dest: int,
        comm: xcclContext_t,
        stream: xpuStream_t,
    ) -> None:
        self.XCCL_CHECK(
            self._funcs["_Z9xccl_sendPvPKvmi12XCCLDataTypeS_"](comm, sendbuff, count, dest, datatype, stream)
        )

    def xcclRecv(
        self,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        src: int,
        comm: xcclContext_t,
        stream: xpuStream_t,
    ) -> None:
        self.XCCL_CHECK(
            self._funcs["_Z9xccl_recvPvS_mi12XCCLDataTypeS_"](comm, recvbuff, count, src, datatype, stream)
        )

    def xcclGroupStart(self) -> None:
        self.XCCL_CHECK(self._funcs["_Z16xccl_group_startv"]())
    def xcclGroupEnd(self) -> None:
        self.XCCL_CHECK(self._funcs["_Z14xccl_group_endv"]())

    def xcclBroadcast(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        root: int,
        comm: xcclContext_t,
        stream: xpuStream_t,
    ) -> None:
        self.XCCL_CHECK(
            self._funcs["_Z14xccl_broadcastPvPKvS_m12XCCLDataTypeiS_"](
                comm, sendbuff, recvbuff, count, datatype, root, stream
            )
        )

    def xcclCommDestroy(self, comm: xcclContext_t) -> None:
        self.XCCL_CHECK(self._funcs["xccl_destroy_context"](comm))

    # ------------------ new helpers for HA ------------------
    # Note: these functions do not use NCCL_CHECK, as they are intended to be used in error handling path

    def xcclCommGetAsyncError(self, comm: xcclContext_t) -> int:
        # Note: XCCL does not have ncclCommGetAsyncError
        raise NotImplementedError("XCCL does not support async error query.")


    def xcclCommAbort(self, comm: xcclContext_t) -> None:
        # abort can return error itself; ignore it to avoid masking original issue
        self._funcs["xccl_comm_abort"](comm)


class xcclResultEnum:
    """Enumeration of XCCL result codes from xccl.h."""

    # XCCL error codes
    xcclSuccess = 0  # XCCL_SUCCESS
    xcclRuntimeError = 1  # XCCL_RUNTIME_ERROR # NV: ncclUnhandledCudaError = 1 
    xcclSystemError = 2  # XCCL_SYSTEM_ERROR
    xcclInternalError = 3  # XCCL_INTERNAL_ERROR
    xcclInvalidArgument = 4  # XCCL_INVALID_ARGUMENT
    # ncclInProgress = 7
    # ncclInvalidUsage = 5
    # ncclRemoteError = 6
    # ncclInProgress = 7
    # Note: XCCL does not have equivalents for ncclInvalidUsage, ncclRemoteError, ncclInProgress

    @staticmethod
    def is_ok(result: int) -> bool:
        """Return True if *result* represents a non-error condition."""
        # Note: XCCL does not have equivalents for ncclInvalidUsage, ncclRemoteError, ncclInProgress
        return result == xcclResultEnum.xcclSuccess


__all__ = [
    "XCCLLibrary",
    "xcclDataTypeEnum",
    "xcclRedOpTypeEnum",
    "xcclResultEnum",
    "xcclUniqueId",
    "xcclContext_t",
    "xpuStream_t",
    "buffer_type",
    "ncclConfig_t",
]
