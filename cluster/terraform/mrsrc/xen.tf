variable "USER" {
  type = string
}

variable "PASSWORD" {
  type = string
}

variable "XO_IP" {
  type = string
}

variable "VMS" {
  type = number
}

variable "KUBE_CONTROL_HOSTS" {
  type = number
}

terraform {
  required_providers {
    xenorchestra = {
      source = "terra-farm/xenorchestra"
      version = "~> 0.9"
    }
  }
}

provider "xenorchestra" {
  url      = "ws://${var.XO_IP}"
  username = var.USER
  password = var.PASSWORD
}

data "xenorchestra_pool" "pool" {
  name_label = "jarana"
}

data "xenorchestra_template" "template" {
  name_label = "cluster_vm"
}

data "xenorchestra_sr" "sr" {
  name_label = "Local storage"
  pool_id = data.xenorchestra_pool.pool.id
}

data "xenorchestra_network" "network" {
  name_label = "Pool-wide network associated with eth3"
  pool_id = data.xenorchestra_pool.pool.id
}

resource "xenorchestra_vm" "vms" {
  count = var.VMS
  memory_max = count.index < var.KUBE_CONTROL_HOSTS ? 12884901888 : 21474836480
  cpus = count.index < var.KUBE_CONTROL_HOSTS ? 4 : 6
  name_label = "node${count.index+1}"
  template = data.xenorchestra_template.template.id
  cloud_config = templatefile("../cloud_config/cloud_config.tftpl", {
    hostname = "node${count.index+1}"
  })
  cloud_network_config = templatefile("../cloud_config/cloud_network_config.tftpl", {
    ip = "10.15.201.${count.index+1}"
  })
  network {
    network_id = data.xenorchestra_network.network.id
  }
  disk {
    sr_id = data.xenorchestra_sr.sr.id
    name_label = "VM root volume"
    size = count.index < var.KUBE_CONTROL_HOSTS ? 53687091200 : 536870912000
  }
}