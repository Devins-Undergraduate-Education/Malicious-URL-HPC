import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import paramiko
import os
import subprocess
import socket
from dotenv import load_dotenv
import time

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
load_dotenv()

def check_vpn_connection():
    """Check if connected to Georgia Tech VPN"""
    try:
        socket.gethostbyname('login-ice.pace.gatech.edu')
        return True
    except socket.gaierror:
        return False

def verify_ice_credentials(username, password):
    """Verify ICE credentials by attempting SSH connection"""
    if not check_vpn_connection():
        return False, "Not connected to Georgia Tech VPN. Please connect to GlobalProtect VPN first."
    
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect('login-ice.pace.gatech.edu', 
                      username=username, 
                      password=password,
                      timeout=10)
        
        # Test if we can execute a simple command
        stdin, stdout, stderr = client.exec_command('echo "Connection test"')
        if stdout.channel.recv_exit_status() == 0:
            client.close()
            return True, None
        else:
            client.close()
            return False, "Could not execute commands on ICE cluster"
    except paramiko.AuthenticationException:
        return False, "Invalid credentials"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def execute_ice_command(command):
    """Execute a command on ICE cluster and return output"""
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect('login-ice.pace.gatech.edu',
                      username=session['ice_username'],
                      password=session['ice_password'])
        
        stdin, stdout, stderr = client.exec_command(command)
        output = stdout.read().decode()
        error = stderr.read().decode()
        
        client.close()
        return output, error
    except Exception as e:
        return None, str(e)

def setup_hpc_environment():
    """Setup Conda environment and dependencies on ICE HPC."""
    username = session['ice_username']
    hpc_dir = f"/storage/ice1/1/7/{username}/Malicious-URL-HPC"
    
    commands = [
        # Create and navigate to the directory
        f"mkdir -p {hpc_dir}",
        f"cd {hpc_dir}",
        
        # Clone repository using SSH
        "git clone git@github.com:dfromond3/Malicious-URL-HPC.git .",
        
        # Setup conda environment
        "module purge",
        "module load anaconda3/2020.02",
        "source /storage/ice1/1/7/shared/anaconda3/etc/profile.d/conda.sh",
        "conda init bash",
        "source ~/.bashrc",
        "conda create --name hpc_env python=3.8 -y",
        "conda activate hpc_env",
        "conda install -c conda-forge pandas numpy matplotlib seaborn scikit-learn joblib -y",
        "pip install codecarbon"
    ]
    
    for cmd in commands:
        output, error = execute_ice_command(cmd)
        if error and "already exists" not in error.lower():
            return False, f"Failed to setup environment: {error}"
    
    return True, None

@app.route('/')
def index():
    if 'ice_username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['ice_username'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        is_valid, error_message = verify_ice_credentials(username, password)
        
        if is_valid:
            session['ice_username'] = username
            session['ice_password'] = password
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error=error_message)
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save file temporarily
        temp_path = f"uploads/{file.filename}"
        os.makedirs('uploads', exist_ok=True)
        file.save(temp_path)

        # Setup HPC environment
        success, error = setup_hpc_environment()
        if not success:
            raise Exception(error)

        # Upload file to ICE using scp
        hpc_dir = f"/storage/ice1/1/7/{session['ice_username']}/Malicious-URL-HPC"
        scp_command = f"scp {temp_path} {session['ice_username']}@login-ice.pace.gatech.edu:{hpc_dir}/malicious_phish.csv"
        process = subprocess.Popen(scp_command, 
                                 shell=True, 
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        process.communicate(input=f"{session['ice_password']}\n".encode())
        
        if process.returncode != 0:
            raise Exception("Failed to upload file to ICE cluster")
        
        # Upload .slurm script
        with open('submit_job.slurm', 'r') as f:
            slurm_content = f.read()
        
        # Create .slurm file on ICE
        sftp = paramiko.SFTPClient.from_transport(paramiko.Transport(('login-ice.pace.gatech.edu', 22)))
        sftp.connect(username=session['ice_username'], password=session['ice_password'])
        with sftp.open(f"{hpc_dir}/submit_job.slurm", 'w') as f:
            f.write(slurm_content)
        sftp.close()
        
        # Submit job using .slurm script
        output, error = execute_ice_command(f"cd {hpc_dir} && sbatch submit_job.slurm")
        if error:
            raise Exception(f"Failed to submit job: {error}")
        
        job_id = output.strip()
        
        # Clean up
        os.remove(temp_path)

        return jsonify({
            'message': 'File uploaded and job submitted successfully',
            'job_id': job_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status/<job_id>')
def get_job_status(job_id):
    if 'ice_username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
        
    try:
        output, error = execute_ice_command(f"sacct -j {job_id} --format=JobID,State,Start,End")
        if error:
            return jsonify({'error': error}), 500
        return jsonify({'status': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 