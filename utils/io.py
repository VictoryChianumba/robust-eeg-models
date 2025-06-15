def get_motor_event_mapping(event_dict):
    """
    Maps motor imagery events dynamically using the parsed event_dict.
    Returns event_id (for Epochs) and label_map (for label relabeling).
    """
    motor_codes = {
        'left': '769',
        'right': '770',
        'feet': '771',
        'tongue': '772'
    }

    # Ensure all required events are in the event_dict
    missing = [key for key in motor_codes.values() if key not in event_dict]
    if missing:
        raise ValueError(f"Missing expected motor event codes in event_dict: {missing}")

    event_id = {k: event_dict[v] for k, v in motor_codes.items()}
    label_map = {event_dict[v]: i for i, v in enumerate(motor_codes.values())}

    return event_id, label_map
