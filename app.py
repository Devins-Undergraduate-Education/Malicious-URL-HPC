import os
import socket
import logging
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
import paramiko
from dotenv import load_dotenv
from pathlib import PurePosixPath

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax'
)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ICE_HOST = 'login-ice.pace.gatech.edu'
ICE_PORT = 22
PROJECT_REPO = 'https://github.com/Devins-Undergraduate-Education/Malicious-URL-HPC'
BASE_STORAGE = '/storage/ice1/1/7'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
SBATCH_FILENAME = 'submit_job.sbatch'
SBATCH_MEMORY = '4G'
SBATCH_TIME = '1:00:00'
SBATCH_NTASKS = 4

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def check_vpn_connection(host=ICE_HOST, port=ICE_PORT, timeout=5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except socket.error:
        return False


def get_ssh_client(username, password=None, pkey_path=None):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    params = dict(
        hostname=ICE_HOST,
        port=ICE_PORT,
        username=username,
        timeout=10
    )
    if pkey_path:
        params['pkey'] = paramiko.RSAKey.from_private_key_file(pkey_path)
    else:
        params['password'] = password
    client.connect(**params)
    return client


def execute_ice_command(username, password=None, pkey_path=None, command=None):
    client = get_ssh_client(username, password, pkey_path)
    try:
        stdin, stdout, stderr = client.exec_command(f"bash -l -c '{command}'")
        out = stdout.read().decode()
        err = stderr.read().decode()
        return out, err
    finally:
        client.close()


def setup_hpc_environment(username, password=None, pkey_path=None):
    """Clone/pull quietly and set up conda env on ICE."""
    remote_base = str(PurePosixPath(BASE_STORAGE) / username)
    scratch_dir = str(PurePosixPath(BASE_STORAGE) / username / '.conda')
    script = f"""
    set -e
    mkdir -p {remote_base}
    cd {remote_base}
    if [ ! -d Malicious-URL-HPC ]; then
    git clone --quiet {PROJECT_REPO}
    else
    cd Malicious-URL-HPC && git pull -q
    fi
    mkdir -p {scratch_dir} && ln -snf {scratch_dir} ~/.conda
    module purge
    module load anaconda3/2023.03
    eval "$(conda shell.bash hook)"
    conda env list | grep -q hpc_env || conda create --name hpc_env python=3.8 -y
    conda activate hpc_env
    conda install pandas numpy matplotlib seaborn scikit-learn joblib -y
    pip install codecarbon
    """
    out, err = execute_ice_command(username, password, pkey_path, script)
    return (err.strip() == '', err)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    if 'ice_username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['ice_username'])


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('username')
        pwd = request.form.get('password')
        valid, err = setup_hpc_environment(user, pwd)
        if valid:
            session.clear()
            session['ice_username'] = user
            session['ice_password'] = pwd
            return redirect(url_for('index'))
        return render_template('login.html', error=err)
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/upload', methods=['POST'])
def upload_file():
    # Prepare local paths
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    filename = secure_filename(file.filename)
    local_csv = os.path.join(UPLOAD_FOLDER, filename)
    sbatch_local = None        # <-- initialize here
    file.save(local_csv)

    user = session.get('ice_username')
    pwd = session.get('ice_password')
    remote_base = str(PurePosixPath(BASE_STORAGE) / user / 'Malicious-URL-HPC')

    try:
        # ensure environment & repo ready
        success, err = setup_hpc_environment(user, pwd)
        if not success:
            raise RuntimeError(err)

        # upload CSV
        client = get_ssh_client(user, pwd)
        sftp = client.open_sftp()
        remote_csv = str(PurePosixPath(remote_base) / filename)
        sftp.put(local_csv, remote_csv)
        sftp.close(); client.close()

        # write corrected SBATCH script
        sbatch_local = os.path.join(UPLOAD_FOLDER, SBATCH_FILENAME)
        sbatch_content = f"""#!/bin/bash
#SBATCH -J MaliciousURLJob
#SBATCH -N 1
#SBATCH --ntasks-per-node={SBATCH_NTASKS}
#SBATCH --mem-per-cpu={SBATCH_MEMORY}
#SBATCH -t {SBATCH_TIME}
#SBATCH -o output-%j.out
#SBATCH -e error-%j.err

set -e
cd {remote_base}

module purge
module load anaconda3/2023.03
eval "$(conda shell.bash hook)"
conda activate hpc_env

srun python intensive_ml_model_hpc.py {filename}
"""
        with open(sbatch_local, 'w', newline='\n') as fh:
            fh.write(sbatch_content)

        # upload & submit
        client = get_ssh_client(user, pwd)
        sftp = client.open_sftp()
        remote_sbatch = str(PurePosixPath(remote_base) / SBATCH_FILENAME)
        sftp.put(sbatch_local, remote_sbatch)
        sftp.close(); client.close()

        out, err = execute_ice_command(user, pwd, None,
                                       f"cd {remote_base} && sbatch {SBATCH_FILENAME}")
        if err.strip():
            raise RuntimeError(err)
        job_id = out.strip().split()[-1]
        return jsonify({'message': 'Submitted', 'job_id': job_id})

    except Exception as ex:
        logger.exception("Upload failed")
        return jsonify({'error': str(ex)}), 500

    finally:
        # only remove files if they were actually created
        for path in [local_csv, sbatch_local]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


@app.route('/status/<job_id>')
def get_job_status(job_id):
    if 'ice_username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    user = session['ice_username']
    pwd = session['ice_password']
    try:
        out, err = execute_ice_command(user, pwd, None,
                                       f"sacct -j {job_id} --format=JobID,State,Start,End")
        if err.strip():
            raise RuntimeError(err)
        return jsonify({'status': out})
    except Exception as ex:
        logger.exception("Status check failed")
        return jsonify({'error': str(ex)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
