import os
import sys
import tempfile
import webbrowser
import threading
import time

# --- Redirigir stdout/stderr si corren como .exe sin consola (evita crasheos en Windows) ---
if getattr(sys, 'frozen', False):
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w')

from flask import Flask, request, jsonify, send_file, render_template_string
import numpy as np
from netCDF4 import Dataset, num2date
import cftime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Límite de 100MB por archivo

# --- LÓGICA DE CONVERSIÓN UNIFICADA ---
def realizar_conversion(ruta_entrada, ruta_salida):
    with Dataset(ruta_entrada, "r") as src:
        es_corriente = any(v in src.variables for v in ["water_u", "u", "current_u", "eastward_water_velocity"])

        time_var = None
        for posible_nombre in ["time1", "time", "reftime1", "reftime", "time_run"]:
            if posible_nombre in src.variables:
                time_var = src.variables[posible_nombre]
                break
        if time_var is None:
            for var_name in src.variables.keys():
                if "time" in var_name.lower():
                    time_var = src.variables[var_name]
                    break

        if time_var is None:
            raise KeyError("No se encontró la variable de tiempo.")

        lats = src.variables["lat"][:]
        lons = src.variables["lon"][:]
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        flat_lons = lon_grid.flatten()
        flat_lats = lat_grid.flatten()
        ncells_total = len(flat_lons)

        unidades_tiempo = time_var.units
        calendario = getattr(time_var, 'calendar', 'proleptic_gregorian')
        valores_tiempo = time_var[:]
        fechas_reales = num2date(valores_tiempo, units=unidades_tiempo, calendar=calendario)

        epoch = cftime.datetime(1970, 1, 1, calendar=calendario)
        tiempo_segundos = np.array([(dt - epoch).total_seconds() for dt in fechas_reales])
        cant_tiempos = len(tiempo_segundos)

        if es_corriente:
            tipo = "Corrientes"
            u_var_name = None
            v_var_name = None
            for posible_u in ["water_u", "u", "current_u", "eastward_water_velocity"]:
                if posible_u in src.variables:
                    u_var_name = posible_u
                    break
            for posible_v in ["water_v", "v", "current_v", "northward_water_velocity"]:
                if posible_v in src.variables:
                    v_var_name = posible_v
                    break

            if not u_var_name or not v_var_name:
                raise KeyError("No se encontraron las variables de velocidad de corriente.")

            u_data = src.variables[u_var_name][:]
            v_data = src.variables[v_var_name][:]

            if u_data.ndim == 4:
                u_surface = u_data[:, 0, :, :]
                v_surface = v_data[:, 0, :, :]
            elif u_data.ndim == 3:
                u_surface = u_data
                v_surface = v_data
            else:
                raise ValueError(f"Dimensión de datos de corrientes inesperada: {u_data.ndim}D")

            flat_u = u_surface.reshape(cant_tiempos, ncells_total)
            flat_v = v_surface.reshape(cant_tiempos, ncells_total)

            if isinstance(flat_u, np.ma.MaskedArray):
                flat_u = flat_u.filled(0.0)
            if isinstance(flat_v, np.ma.MaskedArray):
                flat_v = flat_v.filled(0.0)

            flat_u_final = flat_u * 1.94384
            flat_v_final = flat_v * 1.94384

        else:
            tipo = "Vientos"
            u_original = src.variables["u-component_of_wind_height_above_ground"][:]
            v_original = src.variables["v-component_of_wind_height_above_ground"][:]

            flat_u_final = u_original.reshape(cant_tiempos, ncells_total)
            flat_v_final = v_original.reshape(cant_tiempos, ncells_total)

    with Dataset(ruta_salida, "w", format="NETCDF3_CLASSIC") as dst:
        dst.createDimension("time", None)
        dst.createDimension("ncells", ncells_total)
        dst.Conventions = "CF-1.0"

        if es_corriente:
            dst.default_view = "U, V"
            dst.data_type = "currents"
            dst.net_class = "Multi-point, static, non-gridded"
            dst.SOURCE = "%Converted from GOODS/HYCOM-like input"
        else:
            dst.default_view = "wind_u, wind_v"
            dst.data_type = "winds"
            dst.net_class = "Multi-point, static, non-gridded"
            dst.SOURCE = "Convertidor Rapido PNA"

        var_lon = dst.createVariable("lon", "f4", ("ncells",))
        var_lon.units = "degrees_east"
        var_lon[:] = flat_lons

        var_lat = dst.createVariable("lat", "f4", ("ncells",))
        var_lat.units = "degrees_north"
        var_lat[:] = flat_lats

        var_ncells = dst.createVariable("ncells", "i4", ("ncells",))
        var_ncells.long_name = "cell number"
        var_ncells[:] = np.arange(ncells_total)

        if es_corriente:
            var_u = dst.createVariable("U", "f4", ("time", "ncells"))
            var_u.long_name = "eastward current"
            var_u.units = "Knots"
            var_u[:] = flat_u_final

            var_v = dst.createVariable("V", "f4", ("time", "ncells"))
            var_v.long_name = "northward current"
            var_v.units = "Knots"
            var_v[:] = flat_v_final
        else:
            var_u = dst.createVariable("wind_u", "f4", ("time", "ncells"))
            var_u.long_name = "eastward wind"
            var_u.units = "m s-1"
            var_u[:] = flat_u_final

            var_v = dst.createVariable("wind_v", "f4", ("time", "ncells"))
            var_v.long_name = "northward wind"
            var_v.units = "m s-1"
            var_v[:] = flat_v_final

        var_fid = dst.createVariable("FID", "i4", ("ncells",))
        var_fid.long_name = "feature id"
        var_fid[:] = np.arange(ncells_total)

        var_time = dst.createVariable("time", "f8", ("time",))
        var_time.units = "seconds since 1970-01-01 00:00:00"
        var_time.calendar = "gregorian"
        var_time.long_name = "time"
        var_time[:] = tiempo_segundos

    return tipo

# --- INTERFAZ HTML ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>STM - Conversor de Vientos y Corrientes</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            width: 550px;
            background-color: #1e293b;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            text-align: left;
            border: 1px solid #334155;
        }
        .header-panel {
            border-bottom: 2px solid #334155;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        .sub-header {
            color: #94a3b8;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin: 0;
            font-weight: 600;
        }
        h1 {
            color: #38bdf8;
            margin: 5px 0 0 0;
            font-size: 1.8rem;
            font-weight: 700;
        }
        p { color: #94a3b8; font-size: 0.95rem; line-height: 1.5; }
        .drop-zone {
            border: 3px dashed #38bdf8;
            padding: 40px 20px;
            border-radius: 8px;
            background-color: #0f172a;
            cursor: pointer;
            transition: 0.2s;
            margin: 25px 0;
            text-align: center;
        }
        .drop-zone:hover, .drop-zone.dragover {
            background-color: #1e293b;
            border-color: #0ea5e9;
        }
        .drop-zone p { margin: 0; font-weight: bold; color: #38bdf8; }
        .btn {
            background-color: #0284c7;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 1rem;
            font-weight: bold;
            border-radius: 6px;
            cursor: pointer;
            transition: 0.2s;
            display: none;
            width: 100%;
            text-align: center;
        }
        .btn:hover { background-color: #0369a1; }
        .btn-reset {
            background-color: #475569;
            margin-top: 10px;
        }
        .btn-reset:hover { background-color: #334155; }
        #status {
            margin-top: 15px;
            font-size: 0.95rem;
            font-weight: bold;
            text-align: center;
        }
        .success { color: #34d399; }
        .error { color: #f87171; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-panel">
            <p class="sub-header">Servicio de Tráfico Marítimo</p>
            <h1>Conversor de Vientos y Corrientes</h1>
        </div>

        <p>Arrastrá tu archivo <b>.nc</b> original aquí abajo. El sistema detectará de forma automática si es un archivo de viento o de corrientes y aplicará los algoritmos correspondientes.</p>

        <div id="drop-zone" class="drop-zone">
            <p id="drop-text">📂 Arrastrá y soltá tu archivo .nc aquí<br><span style="font-size:0.8rem; font-weight:normal; color:#64748b;">o haz clic para buscarlo</span></p>
            <input type="file" id="file-input" accept=".nc" style="display: none;">
        </div>

        <div id="status"></div>

        <button id="btn-convertir" class="btn">Convertir y Descargar</button>
        <button id="btn-reset" class="btn btn-reset">🧹 Limpiar y cargar otro</button>
    </div>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const dropText = document.getElementById('drop-text');
        const btnConvertir = document.getElementById('btn-convertir');
        const btnReset = document.getElementById('btn-reset');
        const statusDiv = document.getElementById('status');
        let fileToUpload = null;

        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (fileInput.files.length) {
                handleFile(fileInput.files[0]);
            }
        });

        function handleFile(file) {
            if (!file.name.endsWith('.nc')) {
                showStatus("Error: Solo se admiten archivos .nc", "error");
                return;
            }
            fileToUpload = file;
            dropText.innerHTML = `📄 <b>${file.name}</b><br><span style="font-size:0.8rem; color:#34d399;">Listo para procesar</span>`;
            btnConvertir.style.display = 'block';
            statusDiv.innerHTML = "";
        }

        function showStatus(text, type) {
            statusDiv.className = type;
            statusDiv.innerText = text;
        }

        btnConvertir.addEventListener('click', async () => {
            if (!fileToUpload) return;

            showStatus("🔄 Procesando archivo...", "success");
            btnConvertir.disabled = true;

            const formData = new FormData();
            formData.append('file', fileToUpload);

            try {
                const response = await fetch('/convertir', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    showStatus("✅ ¡Conversión exitosa!", "success");

                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = fileToUpload.name.replace('.nc', '_convertido.nc');
                    document.body.appendChild(a);
                    a.click();
                    a.remove();

                    btnConvertir.style.display = 'none';
                    btnReset.style.display = 'block';
                } else {
                    const errData = await response.json();
                    showStatus("❌ Error: " + errData.error, "error");
                    btnConvertir.disabled = false;
                }
            } catch (err) {
                showStatus("❌ Error de conexión con el servidor local.", "error");
                btnConvertir.disabled = false;
            }
        });

        btnReset.addEventListener('click', () => {
            fileToUpload = null;
            fileInput.value = "";
            dropText.innerHTML = `📂 Arrastrá y soltá tu archivo .nc aquí<br><span style="font-size:0.8rem; font-weight:normal; color:#64748b;">o haz clic para buscarlo</span>`;
            btnConvertir.style.display = 'none';
            btnConvertir.disabled = false;
            btnReset.style.display = 'none';
            statusDiv.innerHTML = "";
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_PAGE)

@app.route('/convertir', methods=['POST'])
def convertir():
    if 'file' not in request.files:
        return jsonify({"error": "No se subió ningún archivo"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    # Usar carpeta temporal del sistema (segura y con permisos de escritura garantizados)
    tmp_dir = tempfile.mkdtemp(prefix="conversor_pna_")
    temp_in = os.path.join(tmp_dir, "input.nc")
    temp_out = os.path.join(tmp_dir, "output.nc")

    try:
        file.save(temp_in)
        realizar_conversion(temp_in, temp_out)

        nombre_descarga = file.filename.replace(".nc", "_convertido.nc")
        response = send_file(temp_out, as_attachment=True, download_name=nombre_descarga)
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        def limpiar():
            time.sleep(3)
            for f in (temp_in, temp_out):
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass
        threading.Thread(target=limpiar).start()

def abrir_navegador():
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    print("🚀 Iniciando Convertidor Local PNA...")
    threading.Thread(target=abrir_navegador).start()
    app.run(host='127.0.0.1', port=5000, debug=False)
