# kiwi — the /IOTCONNECT app and the calibration app, brought to the i.MX95

Spec: [`work/PILOT-kiwi.md`](../work/PILOT-kiwi.md).
State of play: [`work/STATUS-kiwi.md`](../work/STATUS-kiwi.md).
Started as a copy of `jackal/`, whose [`work/STATUS-jackal.md`](../work/STATUS-jackal.md) the code's section
references still point to.

Two apps on one library. **The calibration app** (`src/calibration/`, its pages in `web/`) calibrates the desk and
the camera and tries them: desk, camera, Fine Tune and Tiles pages, as impala left them. **The /IOTCONNECT app**
(`src/main.py`) takes commands from the dashboard or the console, sends what it found as telemetry, and streams the
camera's picture with the tiles drawn over it over KVS WebRTC. Calibrate first; the two cannot run at once, as each
holds the arm and the camera.

Both read the camera through GStreamer, which comes from the system, not pip: its Python bindings (`gi`) and their
`Gst` and `GstVideo` typelibs, and the base and good plugins. The i.MX95's image has them; on Ubuntu,
`python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 gstreamer1.0-plugins-good`. The venv sees them
(`--system-site-packages`).

On the board, as root:

```
mkdir -p ~/kiwi && cd ~/kiwi
# TODO(owner): the package's download location
wget -O tiles.tgz https://downloads.iotconnect.io/partners/nxp/packages/tiles.tgz
tar zxf tiles.tgz
./install.sh            # once, and again after unpacking a newer package: the arm's USB driver, venv/, packages
./calibrate.sh          # calibrate, then try it: http://<board>:8080
./run.sh                # the app: /IOTCONNECT, the stream, the console
./run.sh --no-iotc      # the app without the cloud: commands at the console
scripts/install-service.sh calibration   # or `app`: that one started at every boot; `none`: neither
```

The package is made on a PC by `scripts/package.sh`. `config/` is not in it, so a newer one unpacked over the old
keeps the calibration and the device's identity.

On a PC, in a venv of your own if you like: `scripts/install.sh`, then `python src/calibration/calibration.py` and
`python src/main.py`.

## /IOTCONNECT

1. Import [`files/device-template.json`](files/device-template.json) as a device template.
2. Create a device with it. Either let /IOTCONNECT make the certificate -- copy the credentials zip's certificate and
   key to `config/device-cert.pem` and `config/device-pkey.pem` -- or paste in `config/device-cert.pem`, which
   `scripts/install.sh` made.
3. Download the device's `iotcDeviceConfig.json` into `config/`.

| command                     | does                                                                                |
| --------------------------- | ----------------------------------------------------------------------------------- |
| `detect`                    | park the arm, take the snapshot, find the tiles and read their letters              |
| `place` *LETTERS*           | detect, then set those letters down along the dropoff arc for the reader; detect    |
|                             | again                                                                               |
| `find-cube` [*red*]         | park the arm, find the red cube in the live view, bring the cup over it             |
| `reader-location` [*where*] | where the reader stands -- `back`, `front`, `left` or `right` -- or, alone, where   |
|                             | they stand now                                                                      |

Each is answered by its acknowledgement, the answer the console prints. The console takes the same commands as text:
`place TEAM`, `find cube`, `reader left`.

| telemetry       | is                                                                                          |
| --------------- | ------------------------------------------------------------------------------------------- |
| `tiles`         | the letters on the tiles detected, left to right from behind the arm, lower case: `?` not   |
|                 | read, `-` blank                                                                             |
| `err_no_reach`  | the letters of those out of the arm's reach; empty when none is                             |
| `err_too_close` | the letters of those too close to another to pick up; empty when none is                    |
| `version`, `sdk_version` | the app's and the SDK's                                                            |

The stream is the camera's live view at the calibration's frame size, made as large as H.264 level 3.1 holds (1104 x
828 from the document camera's 2592 x 1944), with the Tiles page's overlay drawn in -- the areas, the tiles, their
letters, a word being set down -- the red cube after `find-cube`, and a bar: what the app is doing, the last answer,
the cloud and the stream. It is a branch of the camera's GStreamer pipeline, run only while someone watches, at the
live view's 5 frames a second at most: H.264 from the board's hardware encoder where GStreamer has one, else x264 in
PyAV at its fastest, encoded once for every viewer (`src/applib/stream.py`).

## What the two apps share

The calibration's choices carry over: the camera's mode and focus (`config/camera-view.json`, `camera.json`), and the
app's settings, the tiles' height, the grip and the reader (`config/appcfg.json`, only what differs from the
defaults, which are in `src/applib/settings.py`). So does the code: the Tiles page's LOCATE TILES and PLACE are the app's `detect` and
`place` (`src/applib/tilework.py`), Fine Tune's FIND RED CUBE its `find-cube` (`src/applib/redcube.py`), and what both
pages draw over the camera's picture is drawn in Python (`src/applib/overlay/`), the stream's overlay, laid over the
pages' live view as a layer.

| directory           | holds                                                                                    |
| ------------------- | ---------------------------------------------------------------------------------------- |
| `src/main.py`       | the /IOTCONNECT app: the composition root                                                |
| `src/app/`          | what the app does, command by command (`app.py`)                                         |
| `src/applib/`       | how, for both apps: the tiles, the letters, the camera, the arm's moves, the overlay,    |
|                     | the settings; and the app's commands, telemetry, /IOTCONNECT, the stream and WebRTC      |
| `src/maxarm/`       | the arm's library                                                                        |
| `src/calibration/`  | the calibration app and its pages' requests; guide in its `README.md`                    |
| `web/`              | the calibration app's pages: HTML, scripts, styles                                       |
| `config/`           | the calibration, the settings and the device's identity; per rig, not in git             |
| `files/`            | the letter reader, `letters.tflite`, what it was built from, and the device template;    |
|                     | `ch341/<uname -r>/ch341.ko`, the arm's USB serial driver for that i.MX95 image           |
| `training/`         | `letters/`: the reader's recipe                                                          |
| `scripts/`          | `install.sh`, `install-ch341.sh`; `package.sh` makes `tiles.tgz`, `build-ch341.sh` the driver |
|                     | `install-service.sh` starts an app at boot (`systemd/`)                                    |
| `agenttools/`       | the fake board and camera, the overlay drawn on the owner's stills, checks; not shipped  |

The arm's library, its calls and limits: [`src/maxarm/README.md`](src/maxarm/README.md).
