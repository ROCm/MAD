			Installation Notes
		Broadcom Linux User Space RoCE Driver

		      Broadcom Inc.
		 5300 California Avenue
	    	Irvine, California  92617

            Copyright (c) 2016 - 2018 Broadcom Limited
            Copyright (c) 2018 - 2025 Broadcom Inc.
		    All Rights Reserved


Table of Contents
=================

  Introduction
  Driver Dependencies
  Supported rdma-core versions
  Building User space Driver
  Driver Settings
  Driver Defaults
  Unloading and Removing Driver
  Setup Verification
  Configuration Tips

Introduction
============

This file describes the libbnxt_re Linux RoCE user space driver for the
Broadcom NetXtreme-C and NetXtreme-E BCM574xx, BCM575xx and BCM576xx
10/20/25/40/50/100/200/400 Gbps Ethernet Network Controllers.

The next few sections describe on packaging, compiling, and installation.

Driver Dependencies
===================

The user space RoCE driver depends on following kernel modules:

1. Ethernet driver for NetXtreme devices (bnxt_en.ko)
2. RoCE driver for NetXtreme devices (bnxt_re.ko)
3. uVerbs device interface, it is an IB-stack component (ib_uverbs.ko)
4. User space RDMA-CM, it is an IB-stack component (rdma_ucm.ko)

Supported rdma-core versions
============================

Following rdma-core versions are supported with this distribution:

v33 v34 v35 v36 v37 v38 v39
v40 v41 v42 v43 v44 v45 v46 v47 v48
v49 v50 v51 v52 v53 v54 v55 v56

Note: rdma-core version 33 or later is required.

Building User space Driver
==========================

Following are the general guidelines to build and install the driver:

1. Check if rdma-core-devel rpm package is installed. On the  OS'es prior
   to RHEL-7.4 or SLES12-sp3 OR IB-stack supplied from OFED prior to OFED-4.8
   check if libibverbs-devel is installed on the target host.

   # rpm -qa| grep rdma-core-devel
		OR
   # rpm -qa| grep libibverbs-devel (Only on OS'es prior to RHEL-7.4 and SLES12-sp3)

   If the rpm is not installed, then install this rpm and its dependencies
   from the OS distribution disk.

2. Create a directory and extract the files

   # tar xvzf libbnxt_re-<version>.tar.gz

3. Build and install

   # cd libbnxt_re-<version>
   # sh autogen.sh
   # ./configure --sysconfdir=/etc
   # make
   # make install all

Driver Settings
===============

1. Check if bnxt_re.driver file is present in /etc/libibverbs.d. In case
   it is not there then copy bnxt_re.driver file to /etc/libibverbs.d directroy

   # cp bnxt_re.driver /etc/libibverbs.d

2. Edit /etc/ld.so.conf file and append following line to it

   /usr/local/lib

   save and quit the editor and run the command given below

   ldconfig -v


Driver Defaults
===============

Install Path: /usr/local/lib


Unloading and Removing Driver
=============================
To uninstall libbnxt_re, from the source-code path where
the library was built run

   # make uninstall


Setup Verification
==================

This section list the basic commands to verify the user space RoCE driver
configuration on the target host

List RoCE devices
-----------------

# ibv_devices
    device                 node GUID
    ------              ----------------
    bnxt_re1            001018fffead1c91
    bnxt_re0            001018fffead1c90


# ibv_devinfo

hca_id: bnxt_re1
        transport:                      InfiniBand (0)
        node_guid:                      0010:18ff:fead:1c91
        sys_image_guid:                 0010:18ff:fead:1c91
        vendor_id:                      0x14e4
        vendor_part_id:                 5847
        hw_ver:                         0x1405
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: bnxt_re0
        transport:                      InfiniBand (0)
        node_guid:                      0010:18ff:fead:1c90
        sys_image_guid:                 0010:18ff:fead:1c90
        vendor_id:                      0x14e4
        vendor_part_id:                 5847
        hw_ver:                         0x1405
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

Traffic Test
------------

Server: rping -s -d -v -a <ip of server bnxt interface>
For example
rping -s -a 192.172.1.1 -Vv -C 3
server ping data: rdma-ping-0: ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqr
server ping data: rdma-ping-1: BCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrs
server ping data: rdma-ping-2: CDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrst

Client: rping -c -d -v -a <ip of server bnxt interface>
For example
rping -c -a 192.172.1.1 -C 3 -vV
ping data: rdma-ping-0: ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqr
ping data: rdma-ping-1: BCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrs
ping data: rdma-ping-2: CDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrst

Configuration Tips
==================

- On some of the hosts where older version of user space roce driver is
  installed following warning could be flashed while running traffic:

  libibverbs: Warning: couldn't load driver 'bnxt_re':
  libbnxt_re-rdmav2.so: cannot open shared object file: No such file or directory

  The above warning can be resolved by deleting everything in /usr/local/lib
  and deleting bnxtre.driver file from /etc/libibverbs.d. Perform a fresh
  installation of library again following the steps specified in "Building Driver"
  section of this document.

- To install OOB driver on a distro where libbnxt_re is inbox-ed (e.g. SLES12-sp3),
  delete/rename the inbox library from where it's installed(default location:
  /lib64/libibverbs).

- At larger scale when hundreds of  QPs are active and Send/Recv protocol
  is used to exchange data there is a possibility of observing RNR-NAKs. As per
  the IB-specification, RNR-NAks are recoverable errors and the application can
  be tuned to minimize the occurrence of RNR-NAKs. A few parameters which could help
  to minimize the RNR-NAKs
	- Bind the task to the CPU with matching NUMA node to Network adaptor
	- Increase the depth of Rx queue using application specific parameter.
		e.g ib_send_bw has -r option to increase receive queue depth.
	- If the application allows, tune the threshold of completion
	  suppression aka CQ moderation  (e.g ib_send_bw has -Q option)
