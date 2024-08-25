# NCKU_SMILE_Project

1. Download java 
https://www.java.com/en/download/manual.jsp
cd /usr/java
tar zxvf jre-8u421-linux-x64.tar.gz 

2. Install R code
sudo apt update
sudo apt install --no-install-recommends software-properties-common dirmngr
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
sudo apt update
sudo apt install r-base
R --version

3. Install rJava
sudo apt update
sudo apt install openjdk-8-jdk
java -version
sudo R CMD javareconf

4. Install Library in R
install.packages("webchem")
install.packages("fingerprint")
install.packages("rJava")
install.packages("rcdklibs")
install.packages("rcdk")

5. Install some library in python
