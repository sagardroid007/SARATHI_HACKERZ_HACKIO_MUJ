import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
import time

class SarathiDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("Sarathi AI | Smart Driving Companion")
        self.geometry("1200x750")
        ctk.set_appearance_mode("dark")
        
        # Define Premium Color Palette
        self.bg_color = "#0B0E14"       # Deep Void Black
        self.card_color = "#161B22"     # Subtle Grey-Blue
        self.accent_color = "#39FF14"   # Neon Green (Safety)
        self.ai_color = "#00D1FF"       # Cyber Cyan
        self.warning_color = "#FF3131"  # Alert Red

        self.configure(fg_color=self.bg_color)

        # Layout Configuration
        self.grid_columnconfigure(0, weight=1) # Sidebar
        self.grid_columnconfigure(1, weight=4) # Main Feed
        self.grid_rowconfigure(0, weight=1)

        # =================================================================
        # SIDEBAR (LEFT)
        # =================================================================
        self.sidebar = ctk.CTkFrame(self, fg_color="#0D1117", corner_radius=0, border_width=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        
        # Logo with Glow Effect (Simulated)
        self.logo_label = ctk.CTkLabel(self.sidebar, text="SARATHI", 
                                       font=ctk.CTkFont(family="Orbitron", size=32, weight="bold"),
                                       text_color=self.ai_color)
        self.logo_label.pack(pady=(40, 5))
        
        self.sublogo = ctk.CTkLabel(self.sidebar, text="DRIVE INTELLIGENT", 
                                    font=ctk.CTkFont(size=10, weight="bold"),
                                    text_color="gray")
        self.sublogo.pack(pady=(0, 40))

        # --- SYSTEM STATUS SECTION ---
        self.status_container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.status_container.pack(fill="x", padx=20)

        self.create_status_item(self.status_container, "SYSTEM", "ONLINE", self.accent_color)
        
        self.voice_label = self.create_status_item(self.status_container, "VOICE", "STANDBY", "gray")
        self.mode_label = self.create_status_item(self.status_container, "AI MODE", "MONITORING", self.ai_status_color())

        # Separation Line
        self.sep = ctk.CTkFrame(self.sidebar, height=2, fg_color="#30363D", width=180)
        self.sep.pack(pady=30)

        # Quick Actions
        self.quit_btn = ctk.CTkButton(self.sidebar, text="TERMINATE", 
                                      fg_color="transparent", border_width=2, 
                                      border_color=self.warning_color,
                                      text_color=self.warning_color,
                                      hover_color="#2D1212",
                                      command=self.quit)
        self.quit_btn.pack(side="bottom", pady=30, padx=20, fill="x")

        # =================================================================
        # MAIN CONTENT (RIGHT)
        # =================================================================
        self.main_content = ctk.CTkFrame(self, fg_color="transparent")
        self.main_content.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_content.grid_columnconfigure(0, weight=1)
        self.main_content.grid_rowconfigure(0, weight=10) # Video
        self.main_content.grid_rowconfigure(1, weight=1)  # Cards

        # --- VIDEO CONTAINER ---
        self.video_container = ctk.CTkFrame(self.main_content, fg_color=self.card_color, 
                                            border_width=2, border_color="#30363D", corner_radius=15)
        self.video_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.video_container.grid_columnconfigure(0, weight=1)
        self.video_container.grid_rowconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.video_container, text="INITIALIZING CAMERA...", 
                                        font=ctk.CTkFont(size=16))
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # --- BOTTOM INFO CARDS ---
        self.cards_frame = ctk.CTkFrame(self.main_content, fg_color="transparent")
        self.cards_frame.grid(row=1, column=0, sticky="ew", padx=0, pady=10)
        self.cards_frame.grid_columnconfigure((0, 1), weight=1, pad=20)

        # Weather Card
        self.weather_card = self.create_info_card(self.cards_frame, "☁️ WEATHER", "Syncing weather data...")
        self.weather_card.grid(row=0, column=0, padx=(10, 20), sticky="nsew")
        self.weather_text = self.weather_card.winfo_children()[1] # Reference label

        # Navigation Card
        self.nav_card = self.create_info_card(self.cards_frame, "📍 NAVIGATION", "System ready. Say 'Navigate to...'")
        self.nav_card.grid(row=0, column=1, padx=(20, 10), sticky="nsew")
        self.nav_text = self.nav_card.winfo_children()[1] # Reference label

    # --- UI HELPERS ---
    def create_status_item(self, parent, title, value, color):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=10)
        t_lbl = ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=10, weight="bold"), text_color="gray")
        t_lbl.pack(side="left")
        v_lbl = ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=12, weight="bold"), text_color=color)
        v_lbl.pack(side="right")
        return v_lbl

    def ai_status_color(self):
        return self.ai_color

    def create_info_card(self, parent, title, default_text):
        card = ctk.CTkFrame(parent, fg_color=self.card_color, corner_radius=12, border_width=1, border_color="#30363D")
        header = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color="gray")
        header.pack(pady=(15, 5), padx=20, anchor="w")
        content = ctk.CTkLabel(card, text=default_text, font=ctk.CTkFont(size=14), wraplength=250, justify="left")
        content.pack(pady=(0, 20), padx=20, anchor="w")
        return card

    # --- UPDATE METHODS ---
    def update_video(self, frame):
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Smart Resize for Dashboard
            target_h = 480
            aspect = w / h
            target_w = int(target_h * aspect)
            
            img = cv2.resize(img, (target_w, target_h))
            img_pil = Image.fromarray(img)
            img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(target_w, target_h))
            
            self.video_label.configure(image=img_ctk, text="")
            self.video_label.image = img_ctk
        except Exception as e:
            print(f"[GUI ERROR] Video Update Failed: {e}")

    def update_stats(self, voice_on, mode_text, is_drowsy=False):
        if voice_on:
            self.voice_label.configure(text="LISTENING", text_color=self.accent_color)
        else:
            self.voice_label.configure(text="STANDBY", text_color="gray")
            
        if is_drowsy:
            self.mode_label.configure(text="DROWSY!", text_color=self.warning_color)
        else:
            self.mode_label.configure(text=mode_text.upper(), text_color=self.ai_status_color())

    def update_weather(self, text):
        self.weather_text.configure(text=text)

if __name__ == "__main__":
    app = SarathiDashboard()
    app.mainloop()
