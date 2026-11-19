# Remote access: SSH from the laptop to the server

Everything on the server is run over SSH. This page sets up key login, a short `ssh sas` alias
that also opens the dashboard tunnels, and a fixed server address.

**Which machine am I on?** The prompt tells you:

| Prompt | Machine |
|---|---|
| `PS C:\Users\...>` (PowerShell) or `you@laptop ~ %` (macOS) | the **laptop** |
| `<user>@<host>:~$` | the **server**, after `ssh sas` |

Placeholders: `<user>` is your username on the server, `<pc-ip>` is its LAN address,
`<repo-dir>` is where the repo is cloned there (e.g. `~/super-auto-strat`).

## 1. On the server: SSH daemon and its address

```bash
sudo apt-get install -y openssh-server
sudo systemctl enable --now ssh
hostname -I            # the first address is <pc-ip>, e.g. 192.168.1.50
ip link                # the MAC address ("link/ether ...") of that interface, for step 5
```

## 2. On the laptop: a key pair

Windows 10/11 and macOS ship the OpenSSH client. If you already have `~/.ssh/id_ed25519`, skip
this. Otherwise press Enter to accept the path, and set a passphrase if you like:

```
ssh-keygen -t ed25519
```

## 3. Copy the public key to the server

**macOS / Linux:**

```bash
ssh-copy-id <user>@<pc-ip>
```

**Windows (PowerShell)** has no `ssh-copy-id`. This does the same:

```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh <user>@<pc-ip> "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

It asks for the server password one last time. Then `ssh <user>@<pc-ip>` should log in without it.

## 4. A `Host sas` alias with the dashboard tunnels

Add this to the laptop's `~/.ssh/config`. Then `ssh sas` logs in and forwards TensorBoard (6006)
and the Optuna dashboard (8080) to the laptop for as long as the session is open:

```
Host sas
    HostName <pc-ip>
    User <user>
    IdentityFile ~/.ssh/id_ed25519
    LocalForward 6006 localhost:6006
    LocalForward 8080 localhost:8080
    ServerAliveInterval 60
```

**Windows:** write it from PowerShell rather than Notepad, which tends to save it as `config.txt`
(and then SSH silently ignores it):

```powershell
Add-Content -Path $env:USERPROFILE\.ssh\config -Encoding ascii -Value @"

Host sas
    HostName <pc-ip>
    User <user>
    IdentityFile ~/.ssh/id_ed25519
    LocalForward 6006 localhost:6006
    LocalForward 8080 localhost:8080
    ServerAliveInterval 60
"@
```

Check it: `Get-ChildItem $env:USERPROFILE\.ssh` lists `config` with no extension.

Usage:

| Command (laptop) | What it does |
|---|---|
| `ssh sas` | a shell on the server, plus the tunnels |
| `ssh -N sas` | the tunnels only, no shell (leave the window open) |
| `ssh sas "cd <repo-dir> && make ps"` | run one command and come back |
| `scp -r sas:<repo-dir>/runs/<run> runs/` | copy a run to the laptop |

Only the first `ssh sas` can bind the ports. A second session prints
`bind [127.0.0.1]:6006: Address already in use`. That's harmless: the first session's tunnel
still works.

## 5. Keep the server's address fixed: DHCP reservation

In the router's admin page (often http://192.168.1.1 or http://192.168.0.1), find the DHCP / LAN
settings and reserve `<pc-ip>` for the server's MAC address from step 1 ("address reservation",
"static lease"). Otherwise the IP can change after a reboot and `ssh sas` times out.

## 6. Disable password login (after key login works)

Keep your current session open while you do this, and test from a **second** window before
closing it, so a mistake can't lock you out:

```bash
echo -e "PasswordAuthentication no\nKbdInteractiveAuthentication no" | sudo tee /etc/ssh/sshd_config.d/99-keys-only.conf
sudo sshd -t && sudo systemctl restart ssh     # -t: syntax check first
```

From the laptop, `ssh sas` should still work, and
`ssh -o PubkeyAuthentication=no <user>@<pc-ip>` should now fail with `Permission denied (publickey)`.

## Troubleshooting

| Symptom | Meaning | Check |
|---|---|---|
| `Connection timed out` | nothing answers at that address: wrong IP, server off/asleep, laptop on another network, or a firewall dropping packets | `ping <pc-ip>`; on the server `hostname -I`; the DHCP reservation |
| `Connection refused` | the machine is reachable, but nothing listens on port 22 | on the server `sudo systemctl status ssh`, `sudo ufw status` (allow with `sudo ufw allow OpenSSH`) |
| `Permission denied (publickey)` | key not accepted | step 3 again; `ssh -v sas` shows which key it offers |
| asks for a password every time | the key isn't installed, or `~/.ssh` permissions are too open on the server | `chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys` |
| `ssh sas` → `Could not resolve hostname sas` | the config file isn't read | Windows: it must be `config`, not `config.txt` |
| dashboards don't load at localhost:6006 | no tunnel, or the containers are down | keep an `ssh sas` open; on the server `make ps` |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | the server was reinstalled, or the IP now points elsewhere | if expected: `ssh-keygen -R <pc-ip>` |
