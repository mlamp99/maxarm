# About

This demo is a robot arm that finds letter tiles on a desk and spells words with them.
It runs on the NXP FRDM-IMX95 and drives a Hiwonder MaxArm over USB.

A camera looks down at the desk. The board finds the tiles in the picture, reads the letter on each one,
and the arm picks them up with its suction cup and sets them down in a row.

The demo reports to /IOTCONNECT and streams the camera's picture there.


# Hardware

- [NXP FRDM-IMX95](https://www.avnet.com/americas/product/nxp/frdm-imx95/evolve-122131125/)
- [Hiwonder MaxArm](https://www.amazon.com/LewanSoul-Bluetooth-Connection-Programming-Education/dp/B0CG5YJKQV)
It comes with ESP32 controller board, dedicated power supply, and the three cubes (red, green and blue) 
used for camera calibration.
- [Logitech C920s](https://www.amazon.com/Logitech-C920S-Pro-HD-Webcam/dp/B07K986YLL). 
  - While other USB camera may work, they may have undesireable properties like barrel effect, 
  or non-uniform pixels or non-matching aspect ratio properties which could make calibration unusable.
- Blank [25 x 25 mm tiles](https://www.amazon.com/dp/B0CWV4PWZK), with a capital letter written on each.


## Recommended Additional Hardware and Tools

- A dedicated USB-C **fast** phone charger or equivalent to power the NXP board.
- A separate USB-C to USB-C **data cable** will be required to connect the arm to the FRDM-IMX95.
The USB-C to USB-C cable that comes with FRDM-IMX95 can be used instead depending on what power cable you connect to it.
- A webcam stand like this
[InnoGear Webcam Stand](https://www.amazon.com/InnoGear-Webcam-Flexible-Gooseneck-Logitech/dp/B08M5ZGL9R/) or similar,
depening on your desk setuo, is ideal to place and keep the camera in optimal view of the desk.
- If your desk surface is not smooth, packing tape can be applied to the desk below the 
arm's base suction cup to help hold the arm firmly in place. This way we avoid invalidating calibration
or causing the suction pump to fail to pick up a tile.

# Features

- The camera finds the tiles on the desk and a small ML network reads each tile's letter, and which way up it is.
- The arm picks up a tile and sets it down where a word needs it, turned the right way up.
- A calibration web app, run once per setup: the desk, the camera, and a few tests.
- /IOTCONNECT:
  - Commands from the cloud: find the tiles, place a word, find the red cube.
  - Telemetry with the tiles found, and an error when a tile is out of the arm's reach.
  - The camera's picture with the tiles drawn over it, as a WebRTC stream.


# Board Setup

Download LF_v6.18.37-2.1.0_MX95 from the
[NXP Embedded Linux for i.MX Applications Processors](https://www.nxp.com/design/design-center/software/embedded-software/i-mx-software/embedded-linux-for-i-mx-applications-processors:IMXLINUX)
web page, and flash it as the
[Getting Started with FRDM-IMX95](https://www.nxp.com/document/guide/getting-started-with-frdm-imx95:GS-FRDM-IMX95?section=get-software)
guide shows.

* Plug the arm and the camera into the board's USB-A port.
* Connect the arm's power supply and put its switch in the up position.
* Plug the arm's ESP32 board into the USB-C J3 port (USB-C to USB-C required here).


# /IOTCONNECT Setup

### Create an /IOTCONNECT Account

An /IOTCONNECT account with an AWS backend is required. A free trial is available:

* Option #1 **(Recommended)**
/IOTCONNECT via [AWS Marketplace](https://github.com/avnet-iotconnect/avnet-iotconnect.github.io/blob/main/documentation/iotconnect/subscription/iotconnect_aws_marketplace.md) - 60 day trial; AWS account creation required

* Option #2
/IOTCONNECT via [iotconnect.io](https://subscription.iotconnect.io/subscribe?cloud=aws) - 30 day trial; no credit card required

> [!NOTE]
> Be sure to check any SPAM folder for the temporary password after registering.

Log in at [console.iotconnect.io](https://console.iotconnect.io).

### Device Template

Import `files/device-template.json` from the package as a device template:
[Importing a Device Template](https://github.com/avnet-iotconnect/avnet-iotconnect.github.io/blob/main/documentation/iotconnect/import_device_template.md).

### Device

The installer below prints the device's certificate. Then, in /IOTCONNECT:

* Select **Devices** in the left menu, then **Device**, then **Add Device** at the top right.
* Choose a Device Unique ID (for example `imx95-tiles`), an Entity, and the template you imported.
* Choose to upload your own certificate, and paste in the one the installer printed,
including the BEGIN and END lines.
* Click **Save and View**, download `iotcDeviceConfig.json` (the paper and cog icon at the top right),
and copy it to the board:

```bash
scp iotcDeviceConfig.json root@<board's IP>:tiles/config/
```


# Installing the Demo

On the board, as root:

```bash
mkdir -p ~/tiles && cd ~/tiles
# TODO: wget -O tiles.tgz https://downloads.iotconnect.io/partners/nxp/packages/tiles.tgz
# For now, SCP to the boar the tgz built with <pilot-name>/scrpts/package.sh
tar zxf tiles.tgz
./install.sh
```

`install.sh` adds the arm's USB serial driver, which this image does not have,
and installs the Python packages the demo needs into `venv/`.
At the end it prints the device's certificate for /IOTCONNECT (above).

To update, download and unpack a newer package in the same place, and run `./install.sh` again.
Your calibration and the device's identity, in `config/`, are kept.


# Running the Demo

First calibrate, once per setup and again if the camera is moved:

```bash
cd ~/tiles
./calibrate.sh
```

It prints the addresses it can be reached at, like `http://192.168.1.20:8080/`.
Open one in a browser on another machine, and go through the pages from left to right.
Ctrl-C stops it.

Then the demo:

```bash
./run.sh              # with /IOTCONNECT
./run.sh --no-iotc    # without the cloud: type the commands at the console
```

The calibration app and the demo cannot run at the same time: each one needs the arm and the camera.


# Tips and Troubleshooting

- "Arm has no power": the arm's USB cable powers its controller, not its motors.
Connect the arm's power supply and put its switch in the up position.
- No `/dev/ttyUSB0` when the arm is plugged in: Kill all applications. Re-plug the ESP32 USB port.
- The demo picks the first USB camera it finds. Unplug any other USB camera.
