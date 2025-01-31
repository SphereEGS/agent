import os
import time
import threading
from flask import Flask
import smtplib

app = Flask(__name__)
last_ping = None
lock = threading.Lock()

def send_alert():
    smtp_server = os.getenv('SMTP_SERVER')
    smtp_port = int(os.getenv('SMTP_PORT', 587))
    username = os.getenv('SMTP_USER')
    password = os.getenv('SMTP_PASSWORD')
    recipient = os.getenv('ADMIN_EMAIL')

    if not all([smtp_server, smtp_port, username, password, recipient]):
        print("Missing email configuration!")
        return

    message = "Subject: System  Alert."
    
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.sendmail(username, recipient, message)
        print("Alert email sent successfully")
    except Exception as e:
        print(f"Email failed: {str(e)}")

def monitor():
    while True:
        time.sleep(60)  # Check every minute
        with lock:
            if last_ping and (time.time() - last_ping) > 60:
                send_alert()
            elif not last_ping:  # Handle initial state
                send_alert()

@app.route('/ping', methods=['POST'])
def ping():
    global last_ping
    with lock:
        last_ping = time.time()
    return 'System status updated', 200

if __name__ == '__main__':
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
    app.run(host='0.0.0.0', port=5000)
