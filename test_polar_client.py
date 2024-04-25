import subprocess

result_client = subprocess.run(['/root/raymond/courage/la-por/build/bin/client', '/home/data500G/client_config', '/home/data500G/merkle_config','-a'], stdout=subprocess.PIPE, text=True)
# 获取命令的标准输出

output_client = result_client.stdout
print(output_client)