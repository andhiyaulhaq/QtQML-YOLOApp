file(GLOB DLLS "${SOURCE_DIR}/*.dll")
foreach(DLL ${DLLS})
    file(COPY ${DLL} DESTINATION ${DEST_DIR})
endforeach()

# Also copy TBB if it exists
file(GLOB TBB_DLLS "C:/intel/openvino_toolkit/runtime/3rdparty/tbb/bin/*.dll")
foreach(DLL ${TBB_DLLS})
    file(COPY ${DLL} DESTINATION ${DEST_DIR})
endforeach()
