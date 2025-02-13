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
        self.initUI()
        
        # Variáveis de controle
        self.cap = None
        self.video_running = False
        self.model = None
        self.confidence_threshold = 0.5
        self.original_frame_size = (0, 0)
        self.yolo_active = False
        self.frame_count = 0
        self.process_every_n_frames = 1

        self.total_detections = {
            "carro": 0,
            "truck": 0,
            "onibus": 0,
            "van": 0,
            "carreta": 0,
            "reboque": 0,
            "moto": 0,
            "eixo_normal": 0,
            "eixo_suspenso": 0,
            "rodas": 0
        }
        
        # Criar diretório para ROIs se não existir
        self.roi_directory = "video_rois"
        if not os.path.exists(self.roi_directory):
            os.makedirs(self.roi_directory)
        
        # Carregar modelo YOLO
        self.load_model()

    def create_right_panel(self):
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        
        controls_group = QGroupBox("Controles")
        controls_layout = QGridLayout()
        
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
        
        # Grupos de Detecções e Logs permanecem iguais
        detections_group = QGroupBox("Detecções")
        detections_layout = QVBoxLayout()
        self.detections_text = QTextEdit()
        self.detections_text.setReadOnly(True)
        detections_layout.addWidget(self.detections_text)
        detections_group.setLayout(detections_layout)
        layout.addWidget(detections_group)
        
        logs_group = QGroupBox("Logs")
        logs_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        logs_layout.addWidget(self.log_text)
        logs_group.setLayout(logs_layout)
        layout.addWidget(logs_group)
        
        return right_widget

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

    # Resto dos métodos permanecem iguais ao código original
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
          if not hasattr(self.video_label, 'original_width') or self.video_label.original_width == 0:
              self.add_log("Carregue um vídeo ou conecte uma cupdate_frameâmera primeiro!")
              return
          
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

    def toggle_detection(self):
        # Verificar se há fonte de vídeo/câmera
        if self.cap is None:
            self.add_log("Carregue um vídeo ou conecte uma câmera primeiro!")
            return

        if not self.video_running:
            self.video_running = True
            self.start_button.setText("Parar")
            self.timer.start(15)  # Reduzido para 15ms (~66 fps)
            self.add_log("Processamento iniciado")
        else:
            self.video_running = False
            self.start_button.setText("Iniciar")
            if hasattr(self, 'timer'):
                self.timer.stop()
            self.add_log("Processamento parado")

    def reset_counters(self):

        for key in self.total_detections:
            self.total_detections[key] = 0
        self.add_log("Contadores resetados")

    def toggle_detection(self):
        if self.cap is None:
            self.add_log("Carregue um vídeo ou conecte uma câmera primeiro!")
            return

        if not self.video_running:
            # Resetar contadores ao iniciar nova detecção
            self.reset_counters()
            self.video_running = True
            self.start_button.setText("Parar")
            self.timer.start(15)  # ~66 fps
            self.add_log("Processamento iniciado")
        else:
            self.video_running = False
            self.start_button.setText("Iniciar")
            if hasattr(self, 'timer'):
                self.timer.stop()
            self.add_log("Processamento parado")      

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            if isinstance(self.cap, cv2.VideoCapture):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        frame_to_show = frame.copy()
        self.frame_count += 1

        # Mantem as detecções do frame anterior se não existirem
        if not hasattr(self, 'current_detections'):
            self.current_detections = {
                "carro": {"count": 0, "eixos": 0, "eixos_erguidos": 0, "pneus": 0},
                "truck": {"count": 0, "eixos": 0, "eixos_erguidos": 0, "pneus": 0},
                "onibus": {"count": 0, "eixos": 0, "eixos_erguidos": 0, "pneus": 0},
                "moto": {"count": 0, "eixos": 0, "eixos_erguidos": 0, "pneus": 0},
                "van": {"count": 0, "eixos": 0, "eixos_erguidos": 0, "pneus": 0},
                "reboque": {"count": 0, "eixos": 0, "eixos_erguidos": 0, "pneus": 0},
                "bicicleta": {"count": 0, "eixos": 0, "eixos_erguidos": 0, "pneus": 0},
                "carreta": {"count": 0, "eixos": 0, "eixos_erguidos": 0, "pneus": 0}
            }

        # Dicionário para detecções do frame atual
        frame_detections = {
            "carro": {"count": 0, "eixos": 0, "eixos_erguidos": 0, "pneus": 0},
            "truck": {"count": 0, "eixos": 0, "eixos_erguidos": 0, "pneus": 0},
            "onibus": {"count": 0, "eixos": 0, "eixos_erguidos": 0, "pneus": 0},
            "moto": {"count": 0, "eixos": 0, "eixos_erguidos": 0, "pneus": 0},
            "van": {"count": 0, "eixos": 0, "eixos_erguidos": 0, "pneus": 0},
            "reboque": {"count": 0, "eixos": 0, "eixos_erguidos": 0, "pneus": 0},
            "bicicleta": {"count": 0, "eixos": 0, "eixos_erguidos": 0, "pneus": 0},
            "carreta": {"count": 0, "eixos": 0, "eixos_erguidos": 0, "pneus": 0}
        }

        if len(self.video_label.roi_points) > 2:
            roi_points = np.array(self.video_label.roi_points, np.int32)
            cv2.polylines(frame_to_show, [roi_points], True, (255, 0, 0), 2)

            if self.yolo_active and self.frame_count % self.process_every_n_frames == 0:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [roi_points], 255)

                results = self.model.predict(
                    frame,
                    conf=self.confidence_threshold,
                    iou=0.4
                )[0]

                detections_list = []
                current_vehicles = set()  # Para rastrear veículos no frame atual

                # Processamento inicial das detecções
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    classe = results.names[int(box.cls)]
                    conf = float(box.conf)

                    obj_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.rectangle(obj_mask, (x1, y1), (x2, y2), 255, -1)
                    intersecao = cv2.bitwise_and(mask, obj_mask)

                    if np.sum(intersecao) > 0:
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        det_info = {
                            'classe': classe,
                            'bbox': (x1, y1, x2, y2),
                            'center': center,
                            'conf': conf
                        }
                        detections_list.append(det_info)

                        if classe in frame_detections:
                            frame_detections[classe]["count"] += 1
                            current_vehicles.add(classe)

                        color = (0, 255, 0) if classe in ["carro", "van", "moto", "bicicleta"] else \
                            (0, 0, 255) if classe in ["truck", "carreta", "reboque", "onibus"] else \
                            (255, 165, 0)

                        cv2.rectangle(frame_to_show, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame_to_show, f"{classe} {conf:.2f}", 
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Processamento de eixos
                for det in detections_list:
                    if det['classe'] in ['pneu', 'pneus']:
                        min_dist = float('inf')
                        nearest_vehicle = None

                        for other_det in detections_list:
                            if other_det['classe'] in frame_detections:
                                dist = ((det['center'][0] - other_det['center'][0])**2 + 
                                    (det['center'][1] - other_det['center'][1])**2)
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_vehicle = other_det['classe']

                        if nearest_vehicle:
                            if det['classe'] == 'pneu':
                                frame_detections[nearest_vehicle]["eixos"] += 1
                                # Ajuste específico para motos e bicicletas
                                if nearest_vehicle in ['moto', 'bicicleta']:
                                    frame_detections[nearest_vehicle]["pneus"] += 1
                                else:
                                    frame_detections[nearest_vehicle]["pneus"] += 2
                            else:  # pneus = eixo suspenso
                                frame_detections[nearest_vehicle]["eixos"] += 1
                                frame_detections[nearest_vehicle]["eixos_erguidos"] += 1
                                if nearest_vehicle in ['moto', 'bicicleta']:
                                    frame_detections[nearest_vehicle]["pneus"] += 1
                                else:
                                    frame_detections[nearest_vehicle]["pneus"] += 2

                # Verifica veículos que saíram do ROI
                if not hasattr(self, 'previous_vehicles'):
                    self.previous_vehicles = set()
                
                vehicles_left = self.previous_vehicles - current_vehicles
                for veiculo in vehicles_left:
                    if self.current_detections[veiculo]["count"] > 0:
                        # Prepara os dados para salvar
                        detection_info = {
                            "tipo": veiculo,
                            "count": self.current_detections[veiculo]["count"],
                            "eixos": self.current_detections[veiculo]["eixos"],
                            "eixos_erguidos": self.current_detections[veiculo]["eixos_erguidos"],
                            "pneus": self.current_detections[veiculo]["pneus"]
                        }
                        # Salva no JSON antes de limpar
                        self.save_detection_to_json(detection_info)
                        # Só depois limpa as detecções
                        self.current_detections[veiculo] = {"count": 0, "eixos": 0, "eixos_erguidos": 0, "pneus": 0}
                        self.add_log(f"Veículo {veiculo} saiu do ROI - dados salvos")

                # Atualiza o registro de veículos para o próximo frame
                self.previous_vehicles = current_vehicles

                # Atualiza as detecções atuais para os veículos ainda presentes
                for veiculo in frame_detections:
                    if veiculo in current_vehicles:
                        self.current_detections[veiculo]["eixos"] = max(
                            self.current_detections[veiculo]["eixos"],
                            frame_detections[veiculo]["eixos"]
                        )
                        self.current_detections[veiculo]["eixos_erguidos"] = max(
                            self.current_detections[veiculo]["eixos_erguidos"],
                            frame_detections[veiculo]["eixos_erguidos"]
                        )
                        self.current_detections[veiculo]["pneus"] = max(
                            self.current_detections[veiculo]["pneus"],
                            frame_detections[veiculo]["pneus"]
                        )
                        self.current_detections[veiculo]["count"] = frame_detections[veiculo]["count"]

                # Atualização da interface
                y_pos = 30
                padding = 25
                detections_text = "Contagem por Tipo de Veículo:\n\n"

                for veiculo, info in self.current_detections.items():
                    if info["count"] > 0:
                        texto_veiculo = f"{veiculo.capitalize()}: {info['count']}"
                        cv2.putText(frame_to_show, texto_veiculo, (10, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        detections_text += f"{veiculo.capitalize()}: {info['count']}\n"
                        y_pos += padding

                        if info["eixos"] > 0:
                            eixos_normais = info["eixos"] - info["eixos_erguidos"]
                            texto_eixos = f"  Eixos: Total {info['eixos']}"
                            cv2.putText(frame_to_show, texto_eixos, (30, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            detections_text += f"  Eixos: Total {info['eixos']}\n"
                            detections_text += f"  Normais: {eixos_normais}\n"
                            detections_text += f"  Suspensos: {info['eixos_erguidos']}\n"
                            y_pos += padding

                            texto_pneus = f"  Pneus: {info['pneus']}"
                            cv2.putText(frame_to_show, texto_pneus, (30, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            detections_text += f"  Pneus: {info['pneus']}\n"
                            y_pos += padding
                        y_pos += 5
                        detections_text += "\n"

                self.detections_text.setText(detections_text)

        # Converter e exibir frame
        rgb_frame = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def save_detection_to_json(self, veiculo_info):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Criar diretório para detecções se não existir
        detection_dir = "detections"
        if not os.path.exists(detection_dir):
            os.makedirs(detection_dir)
        
        # Nome do arquivo baseado na data
        date_str = datetime.now().strftime("%Y%m%d")
        json_file = os.path.join(detection_dir, f"detections_{date_str}.json")
        
        # Preparar dados da detecção
        detection_data = {
            "timestamp": timestamp,
            "veiculo": veiculo_info["tipo"],
            "contagem": veiculo_info["count"],
            "eixos_total": veiculo_info["eixos"],
            "eixos_normais": veiculo_info["eixos"] - veiculo_info["eixos_erguidos"],
            "eixos_suspensos": veiculo_info["eixos_erguidos"],
            "pneus": veiculo_info["pneus"]
        }
        
        # Carregar dados existentes ou criar novo arquivo
        try:
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {"detections": []}
                
            # Adicionar nova detecção
            data["detections"].append(detection_data)
            
            # Salvar arquivo atualizado
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=4)
                
            self.add_log(f"Detecção salva para {veiculo_info['tipo']}")
            
        except Exception as e:
            self.add_log(f"Erro ao salvar detecção: {str(e)}")

    def update_confidence(self, value):
        self.confidence_threshold = value
        self.add_log(f"Limite de confiança atualizado para: {value:.2f}")

    def add_detection(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.detections_text.append(f"[{timestamp}] {message}")
        # Manter o scroll na última linha
        self.detections_text.verticalScrollBar().setValue(
            self.detections_text.verticalScrollBar().maximum()
        )

    def add_log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Manter o scroll na última linha
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

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