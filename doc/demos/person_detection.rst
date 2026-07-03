.. _demo_person_detection:

Person detection
################

This demo shows real-time person detection from a camera stream, running on the `Axon NPU`_.
It is maintained in the `Person detection demo repository`_, separate from the |EAI| repository.

The demo is an enriched version of the :ref:`Person detection application <app_person_detection>` from |EAI|.
It uses the same hardware, model, and detection pipeline, but adds the following capabilities that are not part of the maintained in the base application:

* Live image and detection streaming to a PC over USB CDC ACM, viewed with a host-side Python script.
* A customized ArduCam Mega driver tuned for a higher capture framerate to support continuous streaming.
* GPIO trace pins for measuring current and duration of each processing phase (capture, preprocessing, inference, postprocessing) with a `Power Profiler Kit II (PPK2)`_.

See the `Person detection demo repository`_ for setup, build instructions, and the streaming viewer.
