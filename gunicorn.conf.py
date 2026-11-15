
def post_worker_init(worker):

    import anomaly_detection

    print(
        "Starting MQTT inside worker:",
        worker.pid,
        flush=True
    )

    anomaly_detection.start_background_services()
