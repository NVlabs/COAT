### Flash Attention Compilation takes too long time

```bash
# https://github.com/Dao-AILab/flash-attention/issues/1038#issuecomment-2439430999
python -m pip install --upgrade pip wheel setuptools
MAX_JOBS=64 python -m pip -v install flash-attn --no-build-isolation

# Compile wheel
MAX_JOBS=64 python -m pip wheel flash-attn --no-build-isolation -v -w ./wheelhouse
pip install ./wheelhouse/flash_attn-xxx.whl
```

### Flash Attention Compilation Failed
Try to install version 2.5.8