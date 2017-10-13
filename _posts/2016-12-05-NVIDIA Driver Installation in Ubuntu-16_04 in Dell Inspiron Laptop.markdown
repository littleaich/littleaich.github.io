---
layout: post
title:  "NVIDIA Driver Installation in Ubuntu-16_04 in Dell Inspiron Laptop"
date:   2016-12-05 22:00:32 -0600
categories: Installations
---
Manually downloading and installing the drivers from the NVIDIA website or open source nouveau drivers might cause the problem of infinite login loop in DELL Inspiron laptops. To avoid this problem, a proprietary driver might be installed from the ubuntu repository following the instructions bellow:

1. If the following command prints anything, the Nouveau drivers are loaded.
```
$ lsmod | grep nouveau
```
2. Blacklist the Nouveau driver as follows :
* Create blacklist-nouveau.conf file in terminal
```
$ sudo gedit /etc/modprobe.d/blacklist-nouveau.conf
```
* Add the following lines [1] :
```
blacklist nouveau
options nouveau modeset=0
```
* Update initramfs :
```
$ sudo update-initramfs -u
```
* Reboot
```
$ sudo reboot now
```
3. Check if NVIDIA driver is installed and uninstall.
* Runfile uninstall :
```
$ sudo /usr/bin/nvidia-uninstall
```
* Deb/RPM uninstall :
```
$ sudo apt-get --purge remove nvidia*
```
4. Update NVIDIA ppa.
```
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt update
```
5. Open Software & Updates --> Additional Drivers --> Using NVIDIA binary driver-version xxx.xx from nvidia-xxx(open-source) --> Apply Changes --> Reboot.
6. Check installed driver version.
```
$ cat /proc/driver/nvidia/version
```

#### __References__
1. <http://www.webupd8.org/2016/06/how-to-install-latest-nvidia-drivers-in.html>
2. [CUDA Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile-nouveau)
