variable "ssh_public_key" {
  type = string
}

variable "USER" {
  type = string
}

variable "PASSWORD" {
  type = string
}

variable "vm_index" {
  type = number
}

terraform {
  required_providers {
    opennebula = {
      source = "OpenNebula/opennebula"
      version = "~> 1.0"
    }
  }
}

provider "opennebula" {
  endpoint      = "http://{OPNIP}/RPC2"
  username      = var.USER
  password      = var.PASSWORD
}

resource "opennebula_virtual_machine" "ubuntu" {
  name        = "k8s-cluster-${var.vm_index}"
  description = "VM"
  cpu         = 2
  vcpu        = 4
  memory      = 10240
  group       = "oneadmin"
  permissions = "777"

  context = {
    SSH_PUBLIC_KEY = var.ssh_public_key
    NETWORK      = "YES"
    START_SCRIPT = "apt update && apt install acl"
    HOSTNAME     = "jarana"
  }

  graphics {
    type   = "VNC"
    listen = "0.0.0.0"
    keymap = "fr"
  }

  os {
    arch = "x86_64"
    boot = "disk0"
  }

  disk {
    image_id = 11
    size     = 60000
    target   = "vda"
  }

  on_disk_change = "RECREATE"

  nic {
    model           = "virtio"
    network_id      = 25
  }

  sched_requirements = "FREE_CPU > 60"
}