import socket

client = None
server = None
isServer = False

def int_to_bytes(dt):
        ba = []
        ba.append(dt & 0xff)
        dt = dt >> 8
        ba.append(dt & 0xff)
        dt = dt >> 8
        ba.append(dt & 0xff)
        dt = dt >> 8
        ba.append(dt & 0xff)
        return bytearray(ba)

def bytes_to_int(dt):
        # PY2
        #return int(''.join(reversed(dt)).encode('hex'), 16)
        # PY3
        return int.from_bytes(dt, byteorder='little', signed=False)

def init_edge():
        global client
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('222.29.98.176', 12581))
        return client

def init_cloud_server():
        global server
        global isServer
        isServer = True
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(('222.29.98.176', 12581))
        server.listen(1)
        return server

def cloud_wait():
        global server
        global client
        while True:
            print("Server waiting")
            client, client_address = server.accept()
            return client

def init_cloud():
        global client
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('127.0.0.1', 12581))
        return client

# packet format: [ payload size (4 bytes) ] [ payload ]

def send(pld):
        payload_len = len(pld)
        client.send(int_to_bytes(payload_len))
        client.send(pld)
        return

def recv():
        global isServer
        if isServer:
            print("Is server")
            cloud_wait()
        len_bytes = client.recv(4)
        #print(len(len_bytes))
        #print("LEN BYTES")
        #print(len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3])
        payload_len = bytes_to_int(len_bytes)
        #print(payload_len)
        payload_buffer = bytes()
        while True:
                received = client.recv(4096)
                payload_buffer += received
                #print(len(payload_buffer))
                payload_len -= len(received)
                if (payload_len <= 0):
                        break;
        return payload_buffer
