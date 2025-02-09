import sys
import cv2
import torch
import numpy as np
from PyQt6.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout, 
                           QWidget, QHBoxLayout, QFileDialog, QMainWindow,
                           QGroupBox, QGridLayout, QScrollArea, QTextEdit,
                           QDoubleSpinBox, QFrame, QLineEdit, QDialog, QFormLayout)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt6.QtCore import QTimer, Qt, QPoint
from ultralytics import YOLO
from datetime import datetime
import json
import os

def check_gpu():
    if torch.cuda.is_available():
        print("Detectando GPU: Sim")
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Detectando GPU: Não")
        print("Usando CPU")

check_gpu()

class CameraInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configurar Câmera PLR")
        self.setGeometry(300, 300, 400, 250)
        
        layout = QFormLayout()
        
        # IP
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("Ex: 192.168.1.100")
        layout.addRow("Endereço IP:", self.ip_input)
        
        # Porta
        self.porta_input = QLineEdit()
        self.porta_input.setPlaceholderText("Ex: 554")
        layout.addRow("Porta:", self.porta_input)
        
        # Protocolo
        self.protocolo_input = QLineEdit()
        self.protocolo_input.setPlaceholderText("rtsp/http")
        layout.addRow("Protocolo:", self.protocolo_input)
        
        # Usuário (opcional)
        self.usuario_input = QLineEdit()
        self.usuario_input.setPlaceholderText("Opcional")
        layout.addRow("Usuário:", self.usuario_input)
        
        # Senha (opcional)
        self.senha_input = QLineEdit()
        self.senha_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.senha_input.setPlaceholderText("Opcional")
        layout.addRow("Senha:", self.senha_input)
        
        # Botões
        btn_layout = QHBoxLayout()
        conectar_btn = QPushButton("Conectar")
        conectar_btn.clicked.connect(self.accept)
        cancelar_btn = QPushButton("Cancelar")
        cancelar_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(conectar_btn)
        btn_layout.addWidget(cancelar_btn)
        
        layout_widget = QWidget()
        layout_widget.setLayout(btn_layout)
        layout.addRow(layout_widget)
        
        self.setLayout(layout)
    
    def get_camera_info(self):
        # Monta a URL completa da câmera
        protocolo = self.protocolo_input.text().lower() or 'rtsp'
        ip = self.ip_input.text()
        porta = self.porta_input.text() or '554'
        usuario = self.usuario_input.text()
        senha = self.senha_input.text()
        
        # Monta a URL com credenciais se fornecidas
        if usuario and senha:
            url = f"{protocolo}://{usuario}:{senha}@{ip}:{porta}/stream"
        else:
            url = f"{protocolo}://{ip}:{porta}/stream"
        
        return url

class VideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.roi_points = []
        self.drawing_roi = False
        self.setMouseTracking(True)
        self.original_width = 0   
        self.original_height = 0  
        self.current_video_path = None  # Para armazenar fonte de vídeo/câmera

    def mousePressEvent(self, event):
        if self.drawing_roi and event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            
            # Obter o tamanho atual do widget e do pixmap
            widget_size = self.size()
            pixmap_size = self.pixmap().size()
            
            # Calcular offsets para centralização
            x_offset = (widget_size.width() - pixmap_size.width()) / 2
            y_offset = (widget_size.height() - pixmap_size.height()) / 2
            
            # Ajustar coordenadas considerando o offset
            x = pos.x() - x_offset
            y = pos.y() - y_offset
            
            # Converter para coordenadas da imagem original
            x = int((x / pixmap_size.width()) * self.original_width)
            y = int((y / pixmap_size.height()) * self.original_height)
            
            # Verificar se o ponto está dentro dos limites da imagem
            if 0 <= x < self.original_width and 0 <= y < self.original_height:
                self.roi_points.append((x, y))
                self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.roi_points and (self.drawing_roi or len(self.roi_points) > 2):
            painter = QPainter(self)
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)

            widget_size = self.size()
            pixmap = self.pixmap()
            if pixmap is None:
                return
                
            pixmap_size = pixmap.size()
            
            x_offset = (widget_size.width() - pixmap_size.width()) / 2
            y_offset = (widget_size.height() - pixmap_size.height()) / 2

            for point in self.roi_points:
                x = (point[0] / self.original_width) * pixmap_size.width() + x_offset
                y = (point[1] / self.original_height) * pixmap_size.height() + y_offset
                painter.drawEllipse(int(x) - 3, int(y) - 3, 6, 6)

            if len(self.roi_points) > 1:
                for i in range(len(self.roi_points) - 1):
                    x1 = (self.roi_points[i][0] / self.original_width) * pixmap_size.width() + x_offset
                    y1 = (self.roi_points[i][1] / self.original_height) * pixmap_size.height() + y_offset
                    x2 = (self.roi_points[i + 1][0] / self.original_width) * pixmap_size.width() + x_offset
                    y2 = (self.roi_points[i + 1][1] / self.original_height) * pixmap_size.height() + y_offset
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))

                if len(self.roi_points) > 2:
                    x1 = (self.roi_points[-1][0] / self.original_width) * pixmap_size.width() + x_offset
                    y1 = (self.roi_points[-1][1] / self.original_height) * pixmap_size.height() + y_offset
                    x2 = (self.roi_points[0][0] / self.original_width) * pixmap_size.width() + x_offset
                    y2 = (self.roi_points[0][1] / self.original_height) * pixmap_size.height() + y_offset
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))

class YOLOApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Variáveis de controle
        self.cap = None
        self.video_running = False
        self.model = None
        self.confidence_threshold = 0.5
        self.original_frame_size = (0, 0)
        self.yolo_active = False
        self.frame_count = 0
        self.process_every_n_frames = 1
        
        # Criar diretório para ROIs se não existir
        self.roi_directory = "video_rois"
        if not os.path.exists(self.roi_directory):
            os.makedirs(self.roi_directory)
            
        # Inicializar a interface
        self.initUI()
        
        # Carregar modelo YOLO depois que a interface está pronta
        self.load_model()

    def create_right_panel(self):
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        
        # Grupo de Controles
        controls_group = QGroupBox("Controles")
        controls_layout = QGridLayout()
        
        # Área de Logs
        log_group = QGroupBox("Logs do Sistema")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        
        # Botão de carregar vídeo
        self.load_video_btn = QPushButton("Carregar Vídeo")
        self.load_video_btn.clicked.connect(self.load_video)
        controls_layout.addWidget(self.load_video_btn, 0, 0)
        
        # Botão de conectar câmera
        self.load_camera_btn = QPushButton("Conectar Câmera IP")
        self.load_camera_btn.clicked.connect(self.load_camera_ip)
        controls_layout.addWidget(self.load_camera_btn, 0, 1)
        
        # Botão de iniciar/parar
        self.start_button = QPushButton("Iniciar")
        self.start_button.clicked.connect(self.toggle_detection)
        controls_layout.addWidget(self.start_button, 1, 0)
        
        # Botões de ROI
        self.roi_button = QPushButton("Definir ROI")
        self.roi_button.clicked.connect(self.toggle_roi)
        controls_layout.addWidget(self.roi_button, 1, 1)
        
        self.clear_roi_button = QPushButton("Limpar ROI")
        self.clear_roi_button.clicked.connect(self.clear_roi)
        controls_layout.addWidget(self.clear_roi_button, 1, 2)
        
        self.save_roi_button = QPushButton("Salvar ROI")
        self.save_roi_button.clicked.connect(self.save_roi)
        controls_layout.addWidget(self.save_roi_button, 1, 3)
        
        # Controle de confiança
        confidence_label = QLabel("Retorno de Confiança:")
        controls_layout.addWidget(confidence_label, 2, 0)
        
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.valueChanged.connect(self.update_confidence)
        controls_layout.addWidget(self.confidence_spin, 2, 1)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Grupo de Detecções 
        detections_group = QGroupBox("Detecções no ROI")
        detections_layout = QVBoxLayout()
        self.detections_text = QTextEdit()
        self.detections_text.setReadOnly(True)
        detections_layout.addWidget(self.detections_text)
        detections_group.setLayout(detections_layout)
        layout.addWidget(detections_group)
        
        # Adicionar área de logs
        layout.addWidget(log_group)
        
        return right_widget

    def toggle_detection(self):
        if self.cap is None:
            self.add_log("Carregue um vídeo ou conecte uma câmera primeiro!")
            return
            
        if not self.video_running:
            # Iniciar processamento
            # if len(self.video_label.roi_points) < 3:
            #     self.add_log("Defina um ROI válido primeiro!")
            #     return
                
            self.video_running = True
            self.yolo_active = True
            self.timer.start(30)  # 30ms = aproximadamente 33 FPS
            self.start_button.setText("Parar")
            self.add_log("Detecção iniciada")
        else:
            # Parar processamento
            self.video_running = False
            self.yolo_active = False
            self.timer.stop()
            self.start_button.setText("Iniciar")
            self.add_log("Detecção parada")
    
    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar Vídeo",
            "",
            "Arquivos de Vídeo (*.mp4 *.avi *.mkv);;Todos os Arquivos (*)"
        )
        
        if file_name:
            # Liberar fonte anterior, se existir
            if self.cap is not None:
                self.cap.release()
            
            # Abrir novo arquivo de vídeo
            self.cap = cv2.VideoCapture(file_name)
            if not self.cap.isOpened():
                self.add_log("Erro ao abrir o vídeo!")
                return
            
            # Definir tamanho do frame
            self.original_frame_size = (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            
            # Atualizar dimensões do label de vídeo
            self.video_label.original_width = self.original_frame_size[0]
            self.video_label.original_height = self.original_frame_size[1]
            
            # Definir caminho atual do vídeo
            self.video_label.current_video_path = file_name
            self.yolo_active = False
            
            # Tentar carregar ROI previamente salvo
            try:
                source_hash = str(hash(file_name))
                roi_file = os.path.join(self.roi_directory, f"roi_{source_hash}.json")
                if os.path.exists(roi_file):
                    with open(roi_file, 'r') as f:
                        roi_data = json.load(f)
                        self.video_label.roi_points = roi_data.get('roi_points', [])
                        if len(self.video_label.roi_points) > 2:
                            self.yolo_active = True
                            self.add_log("ROI carregado e YOLO ativado")
                else:
                    self.video_label.roi_points = []
            except Exception as e:
                self.add_log(f"Erro ao carregar ROI: {str(e)}")
                self.video_label.roi_points = []
                
            self.add_log(f"Vídeo carregado: {file_name}")
            
            # Iniciar timer para processamento de frames
            if not hasattr(self, 'timer'):
                self.timer = QTimer()
                self.timer.timeout.connect(self.update_frame)

    def add_log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Manter o scroll na última linha
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def update_frame(self):
        # Verificar se há fonte de vídeo/câmera
        if self.cap is None or not self.cap.isOpened():
            return

        # Ler frame
        ret, frame = self.cap.read()
        if not ret:
            # Para vídeos, reiniciar
            if isinstance(self.cap, cv2.VideoCapture):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        # Cópia do frame para exibição e processamento
        frame_to_show = frame.copy()
        self.frame_count += 1

        # Sempre desenhar o ROI se existir
        if len(self.video_label.roi_points) > 2:
            roi_points = np.array(self.video_label.roi_points, np.int32)
            cv2.polylines(frame_to_show, [roi_points], True, (255, 0, 0), 2)

        # Processamento YOLO 
        if self.yolo_active and len(self.video_label.roi_points) > 2 and self.frame_count % self.process_every_n_frames == 0:
            # Criar máscara para ROI
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [roi_points], 255)

            # Executar detecção YOLO
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=0.4
            )[0]

            # Preparar texto para exibição na interface
            detections_text = "Veículos no ROI:\n\n"
            vehicle_classes = ["carro", "onibus", "caminhao", "moto", "van", "reboque", "carreta"]

            for classe in vehicle_classes:
                # Variáveis para contagem de veículos e pneus
                vehicle_count = 0
                total_pneus_normais = 0
                total_pneus_erguidos = 0

                # Processar detecções para esta classe
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Criar máscara para o objeto atual
                    objeto_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.rectangle(objeto_mask, (x1, y1), (x2, y2), 255, -1)

                    # Verificar intersecção com o ROI
                    intersecao = cv2.bitwise_and(mask, objeto_mask)
                    
                    # Se houver qualquer intersecção com o ROI
                    if np.sum(intersecao) > 0:
                        box_classe = results.names[int(box.cls)]
                        
                        # Contar veículos
                        if box_classe == classe:
                            vehicle_count += 1
                            color = (0, 255, 0) if classe == "carro" else (0, 0, 255)
                            cv2.rectangle(frame_to_show, (x1, y1), (x2, y2), color, 2)

                        # Contar pneus para este veículo
                        if box_classe == "pneu":
                            total_pneus_normais += 1
                        elif box_classe == "eixo_suspenso":
                            total_pneus_erguidos += 1

                # Adicionar informações se veículo foi detectado
                if vehicle_count > 0:
                    detections_text += f"{classe.capitalize()}:\n"
                    detections_text += f"  Quantidade: {vehicle_count}\n"
                    detections_text += f"  Pneus Normais: {total_pneus_normais}\n"
                    detections_text += f"  Pneus Erguidos: {total_pneus_erguidos}\n\n"

            # Atualizar interface com os dados
            self.detections_text.setText(detections_text)

        # Converter frame para RGB
        rgb_frame = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Redimensionar mantendo proporção
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def load_camera_ip(self):
        dialog = CameraInputDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            camera_url = dialog.get_camera_info()
            
            try:
                # Liberar fonte de vídeo anterior, se existir
                if self.cap is not None:
                    self.cap.release()
                
                # Tentar abrir a câmera IP
                self.cap = cv2.VideoCapture(camera_url)
                
                if not self.cap.isOpened():
                    self.add_log(f"Erro ao conectar à câmera: {camera_url}")
                    return
                
                # Definir informações da fonte
                self.video_label.current_video_path = camera_url
                
                # Obter dimensões do stream
                self.original_frame_size = (
                    int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
                
                self.video_label.original_width = self.original_frame_size[0]
                self.video_label.original_height = self.original_frame_size[1]
                
                self.yolo_active = False
                
                self.add_log(f"Conectado à câmera: {camera_url}")
                
                # Iniciar timer se ainda não existir
                if not hasattr(self, 'timer'):
                    self.timer = QTimer()
                    self.timer.timeout.connect(self.update_frame)
                
            except Exception as e:
                self.add_log(f"Erro ao conectar câmera: {str(e)}")

    def initUI(self):
        self.setWindowTitle("Sistema de Detecção YOLOv8")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=2)
        
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)

    def create_left_panel(self):
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)
        
        self.video_label = VideoLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color: black; }")
        layout.addWidget(self.video_label)
        
        return left_widget

    def toggle_roi(self):
        if not self.video_label.drawing_roi:
            # if not hasattr(self.video_label, 'original_width') or self.video_label.original_width == 0:
            #     self.add_log("Carregue um vídeo ou conecte uma câmera primeiro!")
            #     return
            
            self.video_label.drawing_roi = True
            self.roi_button.setText("Finalizar ROI")
            self.add_log("Modo de desenho ROI iniciado")
        else:
            self.video_label.drawing_roi = False
            self.roi_button.setText("Definir ROI")
            self.add_log("Modo de desenho ROI finalizado")

    def clear_roi(self):
        self.video_label.roi_points = []
        self.yolo_active = False
        self.video_label.update()
        self.add_log("ROI limpo")

    def save_roi(self):
        if len(self.video_label.roi_points) < 3:
            self.add_log("Defina pelo menos 3 pontos para o ROI!")
            return

        try:
            roi_data = {
                "source_width": self.original_frame_size[0],
                "source_height": self.original_frame_size[1],
                "roi_points": self.video_label.roi_points
            }

            # Usar hash da fonte de vídeo (pode ser arquivo ou URL de câmera)
            source_hash = str(hash(self.video_label.current_video_path))
            roi_file = os.path.join(self.roi_directory, f"roi_{source_hash}.json")
            
            with open(roi_file, 'w') as f:
                json.dump(roi_data, f, indent=4)
            
            self.yolo_active = True
            self.add_log(f"ROI salvo e YOLO ativado")

        except Exception as e:
            self.add_log(f"Erro ao salvar ROI: {str(e)}")

    def load_model(self):
        try:
            self.model = YOLO('best.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
            self.add_log("Modelo YOLO carregado com sucesso")
        except Exception as e:
            self.add_log(f"Erro ao carregar modelo YOLO: {str(e)}")
            self.model = None

    def update_confidence(self, value):
        self.confidence_threshold = value
        self.add_log(f"Limite de confiança atualizado para: {value:.2f}")

    def closeEvent(self, event):
        # Liberar recurso de captura
        if self.cap is not None:
            self.cap.release()
        
        # Parar timer
        if hasattr(self, 'timer'):
            self.timer.stop()
        
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
                    