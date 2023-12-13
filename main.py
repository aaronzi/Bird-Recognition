import detection
import identification


def process_image(image):
    """
    Process the uploaded image to detect and identify birds.

    :param image: An image file.
    :return: List of identified bird species.
    """

    # Step 1: Detect Birds in the Image
    detected_birds_images = detection.detect_birds(image)

    # Step 2: Identify Each Detected Bird
    # Assuming identification.identify_bird returns the species of a given bird image
    identified_birds = [identification.identify_bird(bird_image) for bird_image in detected_birds_images]

    return identified_birds
