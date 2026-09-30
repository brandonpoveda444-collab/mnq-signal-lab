# MNQ Signal Lab

Prototipo reproducible para investigar señales intradía de **Micro E-mini Nasdaq-100 (MNQ)**. Entrena offline, infiere en CPU y produce `LONG`, `SHORT` o `NO TRADE`. No conecta con brokers, no envía órdenes y no está diseñado para dinero real.

> No existe una precisión garantizada. Los resultados sintéticos solo comprueban que el software funciona; no aportan evidencia de rentabilidad.

## Diseño

- Datos: generador sintético determinista, CSV OHLCV o histórico CME vía Databento.
- Contratos: selección diaria del vencimiento con mayor volumen y ajuste aditivo de gaps al cambiar el líder. Con Databento se solicita el continuo `MNQ.FUT`; para auditoría profesional conviene descargar contratos individuales y aplicar esta regla.
- Variables causales y vectorizadas: retornos, volatilidad, rango/ATR, tendencia, volumen normalizado, sesión y proxy de spread. No usan barras futuras. Si se añade MBP/MBO, el punto de extensión es `features.py`.
- Etiqueta: dirección del cierre a 5 minutos, con zona neutral de 2 ticks.
- Modelo: `HistGradientBoostingClassifier`, ligero en CPU, seguido de calibración sigmoide en un bloque temporal posterior al entrenamiento.
- Validación: 20% final fuera de muestra, recorrido expanding-window en cuatro bloques; antes de cada fold se purgan etiquetas cuyo horizonte cruza el inicio del test.
- Señales: abstención por umbral de probabilidad (0.64 por defecto). El reporte compara precisión, cobertura, expectativa y drawdown para varios umbrales, pero es diagnóstico OOS y **no** debe usarse para reoptimizar el test final.
- Simulación: un MNQ, $2 por punto, comisión, spread y slippage configurables; límite diario, máximo de operaciones, ventana horaria y kill switch.
- Eficiencia: operaciones de características vectorizadas, histogram boosting, caché `pickle` para fuentes externas y tiempos/filas por segundo en el reporte. No requiere GPU.

## Instalación y ejecución

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\mnq-lab --config config.example.yaml
.venv\Scripts\pytest
```

Sin archivo de configuración se usan datos sintéticos y parámetros seguros:

```powershell
.venv\Scripts\mnq-lab
```

Se crean `predictions.csv`, `trades.csv`, `metrics.json` y `config.resolved.json` en `output_dir`. `signal` usa `1=LONG`, `-1=SHORT`, `0=NO TRADE`.

## CSV y Databento

CSV exige `timestamp,open,high,low,close,volume`; acepta además `symbol` y `contract`. Los timestamps se interpretan/normalizan a UTC y luego a la zona configurada. Para Databento:

```powershell
.venv\Scripts\python -m pip install -e ".[databento,dev]"
$env:DATABENTO_API_KEY="..."
# Cambiar data.source a databento y ajustar fechas en config.example.yaml
.venv\Scripts\mnq-lab --config config.example.yaml
```

La caché evita repetir descargas. Bórrala o cambia `cache_path` al modificar símbolo, fechas o esquema; el prototipo no invalida automáticamente claves de caché.

## Interpretación responsable

Optimice en datos de desarrollo por **expectativa neta tras costes y drawdown**, no por accuracy ni cantidad de trades. Mantenga el test final sellado. Precision mide aciertos entre operaciones ejecutadas; coverage mide qué fracción de oportunidades direccionales recibe señal. Un umbral mayor suele elevar precision y reducir coverage, pero no garantiza mejor PnL.

Antes de cualquier paper trading conectado a mercado: validar varios años y regímenes, festivos/DST y sesiones CME; usar contratos individuales y calendario de roll; añadir latencia y fills conservadores; comprobar costes reales del proveedor; hacer nested walk-forward para escoger umbral; paper trade en tiempo real durante varias semanas; revisar drift/calibración. La conexión a broker y las órdenes reales están deliberadamente fuera del proyecto.

## Servicio 24/7 y contenedor

La capa `mnq-service` está separada del motor. Repite el pipeline con intervalo configurable, reintenta con backoff exponencial, publica `/health`, escribe logs JSON y evita alertas duplicadas o demasiado frecuentes. Sin `MNQ_WEBHOOK_URL` solo registra la alerta; con una URL envía JSON por POST. Nunca envía órdenes.

```powershell
Copy-Item .env.example .env
docker build -t mnq-signal-lab .
docker run --rm -p 8080:8080 --env-file .env `
  -v "${PWD}/outputs:/app/outputs" -v "${PWD}/work:/app/work" mnq-signal-lab
Invoke-RestMethod http://127.0.0.1:8080/health
```

Variables documentadas en `.env.example`; no incorpore secretos a la imagen. El ejemplo monta almacenamiento persistente para `outputs/` y `work/`. En el proveedor final, guarde el webhook y la clave Databento en su gestor de secretos, configure el healthcheck como `GET /health` y mantenga una sola réplica mientras la deduplicación sea local. El servicio nunca alerta `NO_TRADE`; solo publica `LONG` o `SHORT`, con deduplicación y separación mínima configurable. Antes de usar un feed en vivo debe sustituirse el ciclo de backtest por un adaptador incremental. El nombre exacto de la plataforma “Hostless” debe confirmarse antes de crear manifiestos específicos o desplegar.

## Alcance de las pruebas

Las pruebas verifican invariancia de variables al añadir futuro (anti-fuga), horizonte de etiqueta, PnL long/short con fricción, límites diarios/horarios, máximo de operaciones, kill switch, rollover, deduplicación/rate limiting y que `NO_TRADE` quede silencioso mientras `LONG`/`SHORT` generan payload de paper signal. No certifican que un feed concreto esté limpio ni que una estrategia sea rentable.
