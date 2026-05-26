import re


def parse_timestamp(timestamp_str: str) -> float:
    """
    Parse timestamp from HH:MM:SS.XX format to seconds
    Args:
        timestamp_str: Timestamp in HH:MM:SS.XX format
    Returns:
        seconds: Time in seconds as float
    Raises:
        ValueError: If timestamp format is invalid
    """
    if not timestamp_str:
        return 0.0
    
    # Regular expression to match HH:MM:SS.XX format
    pattern = r'^(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,2}))?$'
    match = re.match(pattern, timestamp_str)
    
    if not match:
        raise ValueError(
            f"Invalid timestamp format: '{timestamp_str}'. "
            f"Expected format: HH:MM:SS.XX (e.g., 01:23:45.67)"
        )
    
    hours, minutes, seconds, centiseconds = match.groups()
    
    # Validate ranges
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    centiseconds = int(centiseconds or 0)
    
    if minutes >= 60:
        raise ValueError(f"Minutes must be less than 60, got: {minutes}")
    if seconds >= 60:
        raise ValueError(f"Seconds must be less than 60, got: {seconds}")
    if centiseconds >= 100:
        raise ValueError(f"Centiseconds must be less than 100, got: {centiseconds}")
    
    # Convert to total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0
    return total_seconds


def seconds_to_timestamp(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS.XX format
    Args:
        seconds: Time in seconds
    Returns:
        timestamp_str: Formatted timestamp string
    """
    if seconds < 0:
        return "00:00:00.00"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    whole_seconds = int(secs)
    centiseconds = int((secs - whole_seconds) * 100)
    
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def validate_timestamps(start_time: str | None, end_time: str | None, 
                       video_duration: float) -> tuple[float, float]:
    """
    Validate and parse timestamps according to the specified conditions
    Args:
        start_time: Start timestamp string (HH:MM:SS.XX format or None)
        end_time: End timestamp string (HH:MM:SS.XX format or None)
        video_duration: Total video duration in seconds
    Returns:
        start_seconds, end_seconds: Validated timestamps in seconds
    Raises:
        ValueError: If timestamps are invalid or out of bounds
    """
    # Parse start time
    if start_time is None:
        start_seconds = 0.0
    else:
        start_seconds = parse_timestamp(start_time)
    
    # Parse end time
    if end_time is None:
        end_seconds = video_duration
    else:
        end_seconds = parse_timestamp(end_time)
    
    # Validate start time constraints
    if start_seconds < 0:
        raise ValueError("Start time cannot be negative")
    
    if start_seconds >= video_duration:
        raise ValueError(
            f"Start time ({seconds_to_timestamp(start_seconds)}) must be less than video duration "
            f"({seconds_to_timestamp(video_duration)})"
        )
    
    # Validate end time constraints
    if end_seconds <= 0:
        raise ValueError("End time must be greater than 0")
    
    if end_seconds > video_duration:
        raise ValueError(
            f"End time ({seconds_to_timestamp(end_seconds)}) cannot exceed video duration "
            f"({seconds_to_timestamp(video_duration)})"
        )
    
    # Validate start < end relationship
    if start_seconds >= end_seconds:
        raise ValueError(
            f"Start time ({seconds_to_timestamp(start_seconds)}) must be less than "
            f"end time ({seconds_to_timestamp(end_seconds)}). "
            f"Ad placement duration must be positive."
        )
    
    # Additional validation: ensure minimum duration (e.g., at least 2.0 seconds)
    min_duration = 2.0
    if (end_seconds - start_seconds) < min_duration:
        raise ValueError(
            f"Ad placement duration ({seconds_to_timestamp(end_seconds - start_seconds)}) "
            f"is too short. Minimum duration required: {seconds_to_timestamp(min_duration)}"
        )
    
    return start_seconds, end_seconds