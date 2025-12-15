import os

import matplotlib.pyplot as plt
import cv2

class SilentUI:
    def __init__(self, plot_images=False):
        self.components = self.Components()
        self.plot_images = plot_images

    def spinner(self, text):
        return self  # Context manager dummy

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def markdown(self, text, unsafe_allow_html=False):
        if len(text) > 1000:
            text = text[:1000] + '...'
        print(text)

    def image(self, image_path, **kwargs):
        print(f"[Image: {image_path}]")

        if self.plot_images:
            file_name_without_extension = os.path.splitext(os.path.basename(image_path))[0]

            image = cv2.imread(image_path)
            # Convert BGR (OpenCV default) to RGB for matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dpi = 100
            height, width = image_rgb.shape[:2]
            figsize = (width / dpi, height / dpi)
            plt.figure(figsize=figsize, dpi=dpi)
            plt.imshow(image_rgb, cmap='gray', origin='upper')
            plt.axis('off')
            plt.title(file_name_without_extension)
            plt.tight_layout()
            plt.show()

    def code(self, code, language=None):
        print("Code:\n", code)

    def altair_chart(self, chart, use_container_width=False):
        print("[Altair Chart] Rendered chart with Altair")
        try:
            df = chart.data
            print("Chart data preview:")
            print(df.drop(columns="Is Total").to_string(index=False))
        except Exception:
            print("Could not access chart data.")

    class Components:
        def __init__(self):
            self.v1 = self.V1()

        class V1:
            def html(self, html_content, height=None, scrolling=False):
                print("[HTML]")
                print(html_content)