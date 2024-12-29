import flet as ft
from core.auth import AuthManager
from typing import Callable


class AuthView(ft.UserControl):
    def __init__(self, auth_manager: AuthManager, on_success: Callable):
        super().__init__()
        self.auth_manager = auth_manager
        self.on_success = on_success

    def build(self):
        self.email_field = ft.TextField(
            label="Email", border=ft.InputBorder.UNDERLINE, width=300
        )

        self.password_field = ft.TextField(
            label="Password",
            border=ft.InputBorder.UNDERLINE,
            password=True,
            width=300,
        )

        self.error_text = ft.Text(color=ft.colors.RED, size=14, visible=False)

        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        "Spherex Agent", size=40, weight=ft.FontWeight.BOLD
                    ),
                    ft.Text("Sign In", size=20),
                    self.email_field,
                    self.password_field,
                    self.error_text,
                    ft.ElevatedButton(
                        text="Login", on_click=self.handle_login
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=20,
            ),
            alignment=ft.alignment.center,
            expand=True,
        )

    async def handle_login(self, _):
        self.error_text.visible = False
        self.error_text.update()

        success = await self.auth_manager.login(
            self.email_field.value, self.password_field.value
        )

        if success:
            await self.on_success()
        else:
            self.error_text.value = "Invalid credentials"
            self.error_text.visible = True
            self.error_text.update()
