FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
RUN apt update -y
RUN apt upgrade -y
RUN apt install vim openssh-server -y
RUN passwd passwd
RUN echo"PermitRootLogin yes">/etc/ssh/sshd_config
RUN service ssh start
