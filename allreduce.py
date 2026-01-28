"""
In the above script, the allreduce(send, recv) function has a slightly different signature than the ones in PyTorch. It takes a recv tensor and will store the sum of all send tensors in it. As an exercise left to the reader, there is still one difference between our version and the one in DeepSpeech: their implementation divides the gradient tensor into chunks, so as to optimally utilize the communication bandwidth. (Hint: torch.chunk) 
 Implementation of a ring-reduce with addition. """
def allreduce(send, recv):
   rank = dist.get_rank()
   size = dist.get_world_size()
   send_buff = send.clone()
   recv_buff = send.clone()
   accum = send.clone()

   left = ((rank - 1) + size) % size
   right = (rank + 1) % size

   for i in range(size - 1):
       if i % 2 == 0:
           # Send send_buff
           send_req = dist.isend(send_buff, right)
           dist.recv(recv_buff, left)
           accum[:] += recv_buff[:]
       else:
           # Send recv_buff
           send_req = dist.isend(recv_buff, right)
           dist.recv(send_buff, left)
           accum[:] += send_buff[:]
       send_req.wait()
   recv[:] = accum[:] 