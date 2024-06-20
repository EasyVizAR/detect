#!/bin/sh

if [ "$TARGETARCH" = "arm64" ]; then
    fetch_url="https://nvidia.box.com/shared/static/pmsqsiaw4pg9qrbeckcbymho6c01jj4z.whl"
    wheel_name=onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl
    python3 -c "import requests; req = requests.get('$fetch_url', allow_redirects=True); open('$wheel_name', 'wb').write(req.content)"
    python3 -m pip install --no-cache-dir $wheel_name
    rm $wheel_name
else
    python3 -m pip install --no-cache-dir onnxruntime-gpu
fi
