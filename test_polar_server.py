import subprocess
result_server = subprocess.run(['/root/raymond/courage/la-por/build/bin/server', '/home/data500G/server_config', '/home/data500G/merkle_config'], stdout=subprocess.PIPE, text=True)
# 获取命令的标准输出
output_server = result_server.stdout
print(output_server)
