FROM circletree/fyp-ml-fyzr:03-18-2021
RUN apt update -y
RUN apt upgrade -y
RUN apt install vim openssh-server -y
RUN passwd passwd
RUN echo"PermitRootLogin yes">/etc/ssh/sshd_config
RUN service ssh start
