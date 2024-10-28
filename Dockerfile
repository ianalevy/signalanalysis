From rhel9:latest

USER root
RUN dnf install -y python3.12-pip
RUN dnf install -y xcb-util-cursor

RUN dnf install -y zeromq-devel
RUN dnf clean all