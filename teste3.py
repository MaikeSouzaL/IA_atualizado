import cv2
import numpy as np
import json
import datetime
from ultralytics import YOLO

# Carregar modelo YOLO treinado
model = YOLO("best.pt")  # Substituir pelo seu modelo treinado

# Classes do dataset treinado
CLASSES_CUSTOM = [
    "carro", "pneu", "onibus", "caminhao", "moto", "van", "reboque",
    "bicicleta", "carreta", "pneus"
]

# Inicializar variáveis do ROI
roi = None
drawing = False
roi_x1, roi_y1, roi_x2, roi_y2 = -1, -1, -1, -1

# Função para desenhar o ROI com o mouse
def draw_roi(event, x, y, flags, param):
    global roi_x1, roi_y1, roi_x2, roi_y2, drawing, roi

    if event == cv2.EVENT_LBUTTONDOWN:  # Clique inicial
        drawing = True
        roi_x1, roi_y1 = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:  # Arrastar para definir ROI
        roi_x2, roi_y2 = x, y

    elif event == cv2.EVENT_LBUTTONUP:  # Soltar botão do mouse
        drawing = False
        roi = (roi_x1, roi_y1, roi_x2, roi_y2)

# Iniciar captura de vídeo
video_path = "0036.mp4"
cap = cv2.VideoCapture(video_path)

# Criar janela para seleção de ROI
ret, frame = cap.read()
cv2.namedWindow("Selecione a ROI")
cv2.setMouseCallback("Selecione a ROI", draw_roi)

# Exibir vídeo e aguardar seleção da ROI
while True:
    temp_frame = frame.copy()

    # Desenhar o ROI temporário se estiver sendo arrastado
    if roi_x1 != -1 and roi_y1 != -1 and roi_x2 != -1 and roi_y2 != -1:
        cv2.rectangle(temp_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)

    cv2.imshow("Selecione a ROI", temp_frame)

    # Pressione 'ENTER' para confirmar a seleção
    key = cv2.waitKey(1) & 0xFF
    if key == 13 and roi is not None:  # Tecla Enter
        cv2.destroyWindow("Selecione a ROI")
        break

# Criar gravador de vídeo de saída
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output.avi", fourcc, fps, (frame_width, frame_height))

# Lista para rastrear veículos dentro do ROI
rastreamento_veiculos = {}
detections = []

# Loop do vídeo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Sai do loop se o vídeo terminar

    # Desenhar ROI no vídeo
    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2)

    # Rodar YOLO no frame
    results = model(frame)

    # Criar dicionário dinâmico para contagem
    veiculos_detectados = {}

    # Processar resultados
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = CLASSES_CUSTOM[cls]

            # Verificar se o objeto está dentro do ROI
            centro_x = (x1 + x2) // 2
            centro_y = (y1 + y2) // 2

            if roi[1] <= centro_y <= roi[3] and roi[0] <= centro_x <= roi[2]:  # Dentro do ROI
                # Atualizar contagem de veículos e pneus
                if label not in ["pneu", "pneus"]:
                    if label not in rastreamento_veiculos:
                        rastreamento_veiculos[label] = {"quantidade": 0, "pneus": 0, "eixos": 0, "eixo_erguido": 0}
                    rastreamento_veiculos[label]["quantidade"] += 1

                elif label in ["pneu", "pneus"]:
                    for veiculo, dados in rastreamento_veiculos.items():
                        dados["pneus"] += 1

                # Desenhar bounding box e label
                cor = (0, 255, 0)  # Verde
                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

    # Ajustar eixos e detectar eixos erguidos
    veiculos_para_remover = []
    for veiculo, dados in rastreamento_veiculos.items():
        dados["eixos"] = dados["pneus"] // 2
        if dados["pneus"] % 2 != 0:
            dados["eixo_erguido"] += 1

        # Verificar se o veículo saiu do ROI
        if all(y2 < roi[1] for r in results for box in r.boxes for x1, y1, x2, y2 in [map(int, box.xyxy[0])]):
            veiculos_para_remover.append(veiculo)
            detections.append({
                "data_hora": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "veiculo": veiculo,
                "quantidade": dados["quantidade"],
                "eixos": dados["eixos"],
                "pneus": dados["pneus"],
                "eixo_erguido": dados["eixo_erguido"]
            })

    # Remover veículos que saíram do ROI
    for veiculo in veiculos_para_remover:
        del rastreamento_veiculos[veiculo]

    # Exibir informações no canto superior esquerdo
    y_offset = 30
    for veiculo, dados in rastreamento_veiculos.items():
        texto = f"{veiculo.capitalize()}: {dados['quantidade']} | Eixos: {dados['eixos']} | Pneus: {dados['pneus']} | Erguidos: {dados['eixo_erguido']}"
        cv2.putText(frame, texto, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += 30

    # Gravar frame processado no vídeo
    out.write(frame)
    cv2.imshow("Detecção de Veículos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Salvar JSON
with open("detecoes.json", "w") as f:
    json.dump(detections, f, indent=4)

# Liberar recursos
cap.release()

cv2.destroyAllWindows()
