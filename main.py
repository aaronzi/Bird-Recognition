import detection
import identification


def process_image_multi(image):
    """
    Process the uploaded image to detect and identify multiple birds.

    :param image: An image file.
    :return: List of identified bird species.
    """

    # Step 1: Detect Birds in the Image
    detected_birds_images = detection.detect_birds(image)

    # Step 2: Identify Each Detected Bird
    identified_birds = [identification.identify_bird(bird_image) for bird_image in detected_birds_images]

    return identified_birds


def process_image_single(image):
    """
    Process the uploaded image to detect and identify a single bird.

    :param image: An image file.
    :return: Identified bird species.
    """

    # Step 1: convert image to numpy array
    image = detection.read_image(image)

    # Step 2: Identify Bird
    identified_bird = identification.identify_bird(image)

    return identified_bird
