import flet as ft
from core.auth import AuthManager
from ui.auth import AuthView
from ui import MainView
from core.model import ModelManager
from core.camera import CameraManager


class SpherexApp:
    def __init__(self):
        self.auth_manager = AuthManager()
        self.model_manager = ModelManager()
        self.camera_manager = CameraManager()

    async def initialize(self, page: ft.Page):
        self.page = page
        self.page.title = "Spherex Agent"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.padding = 0
        self.page.window_width = 1000
        self.page.window_height = 800
        self.page.window_resizable = False

        # Check authentication status
        if not self.auth_manager.is_authenticated():
            auth_view = AuthView(self.auth_manager, self.on_auth_success)
            await self.page.add_async(auth_view)
        else:
            await self.show_main_view()

    async def on_auth_success(self):
        await self.page.clean_async()
        await self.show_main_view()

    async def show_main_view(self):
        # Load model if not already loaded
        loading_progress = ft.ProgressBar(width=400)
        await self.page.add_async(
            ft.Column(
                [ft.Text("Loading model...", size=20), loading_progress],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            )
        )

        await self.model_manager.load_model()
        await self.page.clean_async()

        main_view = MainView(self.model_manager, self.camera_manager)
        await self.page.add_async(main_view)


async def main(page: ft.Page):
    app = SpherexApp()
    await app.initialize(page)


if __name__ == "__main__":
    ft.app(target=main)
