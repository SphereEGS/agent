import flet as ft
import cv2
from core.model import ModelManager
from  core.camera import CameraManager

class MainView(ft.UserControl):
    def __init__(
        self,
        model_manager: ModelManager,
        camera_manager: CameraManager
    ):
        super().__init__()
        self.model_manager = model_manager
        self.camera_manager = camera_manager
        
    def build(self):
        self.video_view = ft.Image(
            width=800,
            height=600,
            fit=ft.ImageFit.CONTAIN
        )
        
        self.detection_info = ft.Text(size=16)
        
        return ft.Row(
            [
                ft.Column(
                    [
                        self.video_view
                    ],
                    expand=True
                ),
                ft.Column(
                    [
                        ft.Text("Detection Results", size=20, weight=ft.FontWeight.BOLD),
                        self.detection_info
                    ],
                    width=300
                )
            ],
            expand=True
        )
        
    def did_mount(self):
        self.camera_manager.start()
        self.update_timer = self.page.timer(1/30, self.update_frame)
        
    def will_unmount(self):
        self.update_timer.cancel()
        self.camera_manager.cleanup()
        
    def update_frame(self):
        frame = self.camera_manager.get_frame()
        if frame is not None:
            # Process frame with model
            results = self.model_manager.model(frame)
            
            # Convert frame to image for display
            _, buffer = cv2.imencode('.jpg', frame)
            self.video_view.src_base64 = buffer.tobytes()
            
            # Update detection info
            if len(results) > 0:
                # Add detection result processing here
                self.detection_info.value = "License plate detected!"
            else:
                self.detection_info.value = "No detection"
                
            self.update()