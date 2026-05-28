#!/usr/bin/env bash
#
# git_ssh_helpers.sh
# !! Place this file into C:\Program Files\Git\etc\profile.d !!
# Provides gitsetup() and gitlogin() for SSH key workflows

# Generate an ED25519 key, copy the public key to Windows clipboard,
# echo the public key text, and open Github (for example).
gitsetup() {
    if [ -z "$1" ]; then
        echo "Usage: gitsetup <username> <domain>"
        return 1
    fi
    if [ -z "$2" ]; then
        echo "Usage: gitsetup <username> <domain>"
        return 1
    fi
    local name="$1"
    local host="$2"
    local keydir="${HOME}/.ssh/"
    mkdir -p "${keydir}"

    ssh-keygen -t ed25519 -b 4096 \
        -C "${name}@${host}" \
        -f "${keydir}/${name}"

    # Copy to Windows clipboard
    cat "${keydir}/${name}.pub" | clip.exe
    echo
    echo "✓ Public key copied to clipboard."
    echo

    # Echo it in the terminal too
    echo "----- BEGIN PUBLIC KEY (${name}.pub) -----"
    cat "${keydir}/${name}.pub"
    echo "-----  END PUBLIC KEY (${name}.pub)  -----"
    echo
    echo "Go to your host",
    echo " click “Add key” and paste (or re-copy) the above."

    # Open in default browser (Windows)
    explorer.exe "https://github.com/settings/keys"
}

# Start ssh-agent (if not already), and add the named key
gitlogin() {
    if [ -z "$1" ]; then
        echo "Usage: gitlogin <username>"
        return 1
    fi
    local name="$1"
    local keyfile="${HOME}/.ssh/${name}"
    if [ ! -f "${keyfile}" ]; then
        echo "Key not found: $keyfile"
        echo "Run: gitsetup ${name} <domain>"
        return 1
    fi

    # Start agent if needed
    if [ -z "$SSH_AGENT_PID" ]; then
        eval "$(ssh-agent -s)"
    fi

    ssh-add -t 43200 "${keyfile}" # 12 hours
}

alias gitsetup=gitsetup
alias gitlogin=gitlogin
alias gitlogon=gitlogin  # redundancy :)
