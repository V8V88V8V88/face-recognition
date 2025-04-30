import gi
import csv
import subprocess
import os

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

class LoginWindow(Gtk.Window):
    def __init__(self, app):
        super().__init__(title="Login", application=app)
        self.set_default_size(300, 200)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.set_child(box)

        username_label = Gtk.Label(label="Username:")
        self.username_entry = Gtk.Entry()
        password_label = Gtk.Label(label="Password:")
        self.password_entry = Gtk.Entry()
        self.password_entry.set_visibility(False)

        login_button = Gtk.Button(label="Login")
        login_button.connect("clicked", self.on_login_clicked)

        box.append(username_label)
        box.append(self.username_entry)
        box.append(password_label)
        box.append(self.password_entry)
        box.append(login_button)

    def on_login_clicked(self, button):
        username = self.username_entry.get_text()
        password = self.password_entry.get_text()

        # Read credentials from CSV file
        with open("credentials.csv", "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2 and row[0] == username and row[1] == password:
                    self.destroy()
                    app = Gtk.Application.get_default()
                    app_window = MyWindow(application=app)
                    app_window.present()
                    return

        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Invalid username or password",
        )
        dialog.present()

class MyWindow(Gtk.ApplicationWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_title("Run Python Program")
        self.set_default_size(800, 600)

        grid = Gtk.Grid(column_homogeneous=True, row_homogeneous=True, column_spacing=20, row_spacing=20)
        self.set_child(grid)

        welcome_label = Gtk.Label(label="Welcome to Face Recognition System By TheFOSSClub")
        grid.attach(welcome_label, 0, 0, 2, 1)

        self.run_button = Gtk.Button(label="Run face2.py")
        self.run_button.connect("clicked", self.open_file)
        grid.attach(self.run_button, 0, 1, 1, 1)

        self.exit_button = Gtk.Button(label="Exit")
        self.exit_button.connect("clicked", self.on_exit_clicked)
        grid.attach(self.exit_button, 1, 1, 1, 1)

        self.run_button.set_size_request(200, 50)
        self.exit_button.set_size_request(200, 50)

        self.set_child(grid)

    def open_file(self, widget):
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face2.py")
        subprocess.Popen(['python3', file_path])

    def on_exit_clicked(self, widget):
        self.get_application().quit()

class LoginApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.example.loginapp")

    def do_activate(self):
        login_window = LoginWindow(self)
        login_window.present()

if __name__ == "__main__":
    app = LoginApp()
    app.run(None)

