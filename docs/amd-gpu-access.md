# AMD GPU Access Notes

Use this note when a future agent needs to verify access to the hackathon AMD
GPU host. Do not print secrets from `.env` in logs or chat.

## Connection Source

The project-local `.env` contains the AMD host credentials:

```bash
AMD_IP=
AMD_SSH_USER=root
AMD_ROOT_PASSWORD=
AMD_JUPYTER_URL=
AMD_JUPYTER_TOKEN=
```

The committed `.env.example` lists the expected keys, but real values stay only
in the uncommitted `.env`.

## Verified Connection

Connection was tested from Codex on 2026-06-27 using the `.env` values.

Observed remote host:

- Hostname: `tumai-paris-hackathon-6`
- Kernel: `Linux 6.8.0-124-generic x86_64 GNU/Linux`
- SSH user: `root`
- ROCm tool available: `/usr/bin/rocm-smi`
- Python available: `/usr/bin/python3`

Observed GPU:

- Device: `AMD Instinct MI300X VF`
- GFX target: `gfx942`
- VRAM total: `205822885888` bytes
- ROCm driver version reported by `rocm-smi`: `6.16.13`

System Python did not have PyTorch installed at the time of the check.

## Safe Connection Test

Future agents can test SSH and GPU visibility with a command like this. It uses
the password from `.env` without echoing it.

```bash
set -a
. ./.env
set +a

expect <<'EOF'
set timeout 60
set host $env(AMD_IP)
set user $env(AMD_SSH_USER)
set pass $env(AMD_ROOT_PASSWORD)

spawn ssh \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/tmp/ehl_amd_known_hosts \
  -o ConnectTimeout=15 \
  $user@$host "hostname; uname -srmo; rocm-smi --showproductname --showdriverversion --showmeminfo vram"

expect {
  -re "(?i)password:" { send "$pass\r"; exp_continue }
  eof
}
catch wait result
exit [lindex $result 3]
EOF
```

## Current Training Caveat

The connection and AMD GPU visibility are confirmed. Full model training has not
yet been set up on the remote host.

At the time of the check:

- `python3 -m venv` failed because `python3.12-venv` was missing.
- PyTorch was not installed in system Python.
- No project checkout or existing virtual environment was found under `/root` or
  `/workspace`.

When training work begins, create an isolated remote workspace, install a ROCm
PyTorch build, then verify PyTorch sees the MI300X through:

```python
import torch

print(torch.__version__)
print(torch.version.hip)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
```
