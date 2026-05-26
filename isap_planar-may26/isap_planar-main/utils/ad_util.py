import numpy as np
from PIL import Image, ImageSequence


class AdManager:
    """
    Class to manage advertisement templates for in-scene ad placement.
    It supports loading static images and GIFs for ad placement.
    The class handles the setup of advertisement frames, durations, and cycle durations.
    """

    def __init__(self):
        """
        Initialize the AdManager with the advertisement file.
        """
        self.ad_frames = []
        self.ad_durations = []
        self.ad_frame_count = 0
        self.ad_frame_time = []
        self.ad_cycle_duration = 0.0

    def setup_ad_template(self, ad_file):
        """
        Setup the advertisement template from the provided file.
        Args:
            ad_file (str): Path to the advertisement image file.
        Raises:
            ValueError: If the advertisement file format is unsupported.
        """
        print(f"\nSetting up advertisement template from: {ad_file}...")

        # check if ad_file is png, jpg or jpeg
        if ad_file.split(".")[-1].lower() in ["png", "jpg", "jpeg"]:
            print("Loading advertisement image...")

            # load a static image for ad placement
            ad = Image.open(ad_file)
            self.ad_frames = [
                frame.copy().convert("RGBA") for frame in ImageSequence.Iterator(ad)
            ]
            self.ad_durations = [
                frame.info.get("duration", 100) for frame in ImageSequence.Iterator(ad)
            ]

            # Apply frame around each ad image
            # self.ad_frames = self._add_frame_around_ad(self.ad_frames)
            self.ad_frame_time = np.cumsum(self.ad_durations) / 1000.0
            self.ad_cycle_duration = self.ad_frame_time[-1]

        # if ad_file is a gif
        elif ad_file.split(".")[-1].lower() in ["gif"]:
            print("Loading advertisement GIF...")

            # load a GIF for ad placement
            ad = Image.open(ad_file)
            self.ad_frames = [
                frame.copy().convert("RGBA") for frame in ImageSequence.Iterator(ad)
            ]
            self.ad_durations = [
                frame.info.get("duration", 100) for frame in ImageSequence.Iterator(ad)
            ]

            # Apply frame around each ad image
            # self.ad_frames = self._add_frame_around_ad(self.ad_frames)
            self.ad_frame_time = np.cumsum(self.ad_durations) / 1000.0
            self.ad_cycle_duration = self.ad_frame_time[-1]

        # else
        else:
            raise ValueError(
                f"Unsupported advertisement file format: {ad_file.split('.')[-1]}"
            )

        print(f"advertisement template loaded!\n\n")
        

    def _add_frame_around_ad(
        self,
        frames,
        frame_thickness=None,
        frame_color=(255, 255, 255, 255),
        inner_padding=None,
        corner_radius=None,
    ):
        """
        Add a decorative frame around each ad image frame.

        Args:
            frames (List[PIL.Image]): List of RGBA images (ad frames).
            frame_thickness (int): Thickness of the frame border in pixels.
            frame_color (tuple): RGBA color of the frame border.
            inner_padding (int): Padding between the ad content and the frame.
            corner_radius (int): Optional rounded corner radius for the frame.

        Returns:
            List[PIL.Image]: New list of frames with the frame applied.
        """
        from PIL import ImageDraw, ImageOps

        if not frames:
            return frames

        framed_frames = []
        for img in frames:
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            w, h = img.size
            # Compute proportional dimensions based on image size
            base = min(w, h)
            t = (
                frame_thickness
                if frame_thickness is not None
                else max(2, int(0.04 * base))
            )
            p = inner_padding if inner_padding is not None else max(1, int(0.02 * base))
            cr = corner_radius if corner_radius is not None else int(0.15 * t)

            # Total added size: frame border on both sides + inner padding
            border = t + p
            new_w, new_h = w + 2 * border, h + 2 * border

            # Create base canvas with transparent background
            canvas = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))

            # Draw the frame shape
            frame_img = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(frame_img)

            # Outer rectangle
            outer_box = [0, 0, new_w - 1, new_h - 1]
            # Inner rectangle where content sits (creates border thickness)
            inner_box = [t, t, new_w - 1 - t, new_h - 1 - t]

            if cr and cr > 0:
                # Rounded rectangle for outer, then punch inner hole by alpha compositing
                # Draw outer rounded rect
                ImageDraw.Draw(frame_img).rounded_rectangle(
                    outer_box, radius=cr, fill=frame_color
                )
                # Draw inner rounded rect as transparent cutout
                cutout = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
                ImageDraw.Draw(cutout).rounded_rectangle(
                    inner_box, radius=max(cr - t, 0), fill=(0, 0, 0, 0)
                )
                # Create a mask to subtract inner area from outer
                mask = Image.new("L", (new_w, new_h), 0)
                ImageDraw.Draw(mask).rounded_rectangle(outer_box, radius=cr, fill=255)
                ImageDraw.Draw(mask).rounded_rectangle(
                    inner_box, radius=max(cr - t, 0), fill=255
                )
                frame_img.putalpha(mask)
            else:
                # Simple rectangular frame: draw outer, then subtract inner area
                draw.rectangle(outer_box, fill=frame_color)
                # Make the inner area transparent by alpha mask
                mask = Image.new("L", (new_w, new_h), 255)
                ImageDraw.Draw(mask).rectangle(inner_box, fill=0)
                frame_img.putalpha(mask)

            # Paste the frame onto canvas
            canvas = Image.alpha_composite(canvas, frame_img)

            # Paste the original image centered within the inner padding area
            paste_pos = (border, border)
            canvas.paste(img, paste_pos, img)

            framed_frames.append(canvas)

        return framed_frames

    def get_ad_frame_for_time(self, t):
        """
        Get the advertisement frame for a given time t.
        Args:
            t (float): Time in seconds.
        Returns:
            PIL.Image: The advertisement frame corresponding to the time t.
        """
        t = t % self.ad_cycle_duration
        for i, ft in enumerate(self.ad_frame_time):
            if t < ft:
                # print(f"ad frame idx: {i}")
                return self.ad_frames[i]

        return self.ad_frames[-1]