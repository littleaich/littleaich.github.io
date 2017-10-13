---
layout: post
title:  "Ubuntu-16_04 Installation in Dell Inspiron Laptop"
date:   2016-12-05 22:00:32 -0600
categories: Installations
---
1. Press F2 to get into the bios.
2. In the boot menu, add boot option --> choose USB disk option.
3. In the USB disk option, choose EFI --> bootx64 EFI and save (F10).
4. In the grub loader boot menu, after highlighting "Install Ubuntu", press "e".
5. Just before the "quiet splash" option, add "ro nomodeset". Here "ro" means read-only.
6. Save (F10) and you will get Ubuntu installing.
7. After installation, blacklist to avoid getting stuck while rebooting or shutting down as follows :

* Open blacklist.conf file in terminal
```
$ sudo gedit /etc/modprobe.d/blacklist.conf
```
* Add the following lines [1] :
```lis
blacklist amd76x_edac
blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist nvidiafb
blacklist rivatv
```
* Update initramfs :
```
$ sudo update-initramfs -u
```
* Reboot
```
$ sudo reboot now
```

#### __References__
1. <https://www.youtube.com/watch?v=pZ-r3gS38RU>
