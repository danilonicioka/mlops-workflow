## Cluster

O cluster não possui uma interface própria, então, para acessá-lo, é preciso utilizar a ferramenta [kubectl](https://kubernetes.io/pt-br/docs/tasks/tools/#kubectl) com o kubeconfig, o qual pode ser encontrado nas variáveis de ambiente deste repositório. Para utilizá-lo, basta copiá-lo para `~/.kube/config` e o comando kubectl conseguirá acessar o cluster.

Outra forma de acessar sem precisar instalar o kubectl ou pegar o kubeconfig é utilizar um dos nodes do próprio cluster por meio do terminal da VM no Xen Orchestra, mas não é recomendado.

Por fim, a plataforma Rancher foi instalada e pode ser utilizada para gerenciar o cluster com uma interface gráfica.

# Cluster K8s Csic

O deploy do cluster k8s é feito com o Kubespray e utiliza as VMs criadas por meio do Terraform e do Xen (XCP-ng e Xen Orchestra). 

Para facilitar esse processo, foi criada uma pipeline para automatizar grande parte das etapas.

# Pipeline

Como o Kubespray será utilizado, antes de iniciar o deploy do cluster, foi criado um job para fazer o build de uma imagem com o Kubespray instalado. Isso porque não há uma imagem oficial com o Kubespray e por causa que a sua instalação é um tanto demorada devido às dependências, o que tornaria o processo de deploy e configuração do cluster mais demorados.

A pipeline atualmente está separada em alguns stages, eles são:
- `build`: Comandos para o build da image docker do kubespray.
- `deploy_cluster`: Processos necessários para o deploy do cluster em si.
- `deploy_apps`: Instalações de ferramentas para o bom funcionamento do cluster.

<!-- >
- `add_node`: Comandos para a criar um novo node no cluster.
- `update_vms`: Comandos para editar uma VM no cluster.
<-->

## Build
### Kubespray

Nessa stage temos apenas o job com o processo de criação dessa imagem docker, a qual consiste, inicialmente, em passar as credenciais do registry para o kaniko com variáveis de ambientes pré-definidas no repositório:

```bash
echo "{\"auths\":{\"${CI_REGISTRY}\":{\"auth\":\"$(printf "%s:%s" "${CI_REGISTRY_USER}" "${CI_REGISTRY_PASSWORD}" | base64 | tr -d '\n')\"}}}" > /kaniko/.docker/config.json
```

Em que:
- `CI_REGISTRY`: endereço do registry onde a imagem será armazenada
- `CI_REGISTRY_USER`: nome de usuário ou do token para acesso a esse registry
- `CI_REGISTRY_PASSWORD`: senha do usuário ou o valor do token indicado na variável anterior

Logo após, informamos o caminho do dockerfile, também via variável de ambiente, e o local para guardar a image identificando a mesma com uma tag de commit:

```bash
/kaniko/executor --context "${CI_PROJECT_DIR}" --dockerfile "${CI_PROJECT_DIR}/Dockerfile" --destination "${CI_REGISTRY_IMAGE}:$CI_COMMIT_TAG"
```
Perceba que o dockerfile é bem simples e realizada a instalação dos pacotes necessários do sistema. A primeira é via apt onde é baixado alguns pacotes já conhecidos como python3 (necessário para o Ansible); e a segunda é o download do arquivos necessários do repositório do kubespray no github a partir do `git clone`.

```bash
apt update && apt install git curl python3 gcc acl -y
```

```bash
git clone https://github.com/kubernetes-sigs/kubespray.git
```

Seguindo, é realizado, em dois comandos, o download e a instalação do gerenciador de pacotes do python chamado `pip`:

```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
```

```bash
python3 get-pip.py --user
```

Preparado os arquivos necessários, pode-se instalar os pacotes solicitados pelo kubespray a partir do seu arquivo de `requirements-2.12.txt`:
```bash
/root/.local/bin/pip3 install -r kubespray/requirements-2.12.txt
```

## Deploy

Stage em que a configuração básica do cluster será feita, desde o deploy das VMs até a configuração com Kubespray.

### Deploy VMs

Job que realiza o deploy das VMs que servirão como os nodes do cluster. 

As VMs são criadas a partir de um template definido no XO para uma VM com Debian 11 inicializável com Cloud Init. Os templates para os cloud configs estão definidos em `terraform/cloud_config`. As configurações das VMs e da infraestrutura estão definidas neste [diagrama](https://app.diagrams.net/#G1WPYQ8qYAPsN_XG4qLA8Bt5E9RwER3LEx#%7B%22pageId%22%3A%22C5RBs43oDa-KdzZeNtuy%22%7D) e especificamente indicados na imagem a seguir:

![cluster_diagram](https://gl.idc.ufpa.br/csic/cluster-k8s-csic/-/raw/main/images/cluster_diagram.png)

O arquivo `terraform/default/xen.tf` é utilizado para referenciar os templates e criar as VMs com as devidas configurações. Além disso, note como algumas variáveis de ambiente são passadas para o terraform: 
- `XO_USER`, `XO_PASSWORD` e `XO_IP` : usuário e senha para autenticar no XO rodando no IP `XO_IP`
- `VMS`: quantidade de VMS para servirem como nodes (masters+workers)
- `KUBE_CONTROL_HOSTS`: quantidade de nodes que serão control planes (masters)

### Cluster

Diferentemente do Terraform, o deploy de um cluster Kubernetes já envolve detalhes mais sotisficados e uma atenção redobrada no momento em que o _job_ está sendo construído. A razão para isso é simples: um cluster k8s possui diversos elementos, então, mesmo a maneira simplificada de realizar a sua implantação ainda irá gerar um trabalho considerável para o administrador. O primeiro desafio é entender o propósito das variáveis de ambiente utilizadas na pipeline. Segue a lista de todas estas:

- `IPS`: Os IPs das máquinas que serão os nós do cluster.
  - Ex: 192.168.56.102 192.168.56.103
- `SSH_PRIVATE_KEY`: Chave privada a ser utilizada para acessar as VMs.
- `SSH_PRIVATE_KEY_FILE`: O mesmo conteúdo de SSH_PRIVATE_KEY, no entanto, formatado como um arquivo.
- `SSH_PUBLIC_KEY`: Chave pública associadas as VMs utilizadas.

Com essas variáveis conhecidas, a melhor maneira de entender todo o processo, é descrevendo as principais linhas da pipeline.

Inicialmente, para facilitar a definição dos IPS das VMS, é utilizado uma combinação de comandos para criar um arquivo chamado `ips` com os IPS das VMS em ordem de acordo com o número de VMs especificado na variável de ambiente `VMS`. Em seguida, esse arquivo é utilizado como fonte para definir o valor da variável de ambiente `IPS`.

```bash
x=1; while [ $x -le $VMS ]; do echo "10.15.201.$x" >> ips && echo $(( x++ )); done
export IPS=`cat ips`
```

Em seguida, é feito um clone do repositório do Kubespray para pegar todos os arquivos necessários para configuração default do cluster. 

```bash
git clone --branch v2.22.1 https://github.com/kubernetes-sigs/kubespray.git
```

Aproveitamos os arquivos disponíveis no diretório `kubespray/inventory/sample`, o qual é dado como exemplo para configuração padrão de um cluster, e copiamos para o diretório que utilizaremos `kubespray/inventory/mycluster`.

```bash
cp -r kubespray/inventory/sample kubespray/inventory/mycluster
```

A partir disso, é necessário organizar o novo diretório com nossos arquivos customizados. Primeiro, cria-se um arquivo de inventário que irá alocar as informações de cada nó no cluster:
```bash
touch kubespray/inventory/mycluster/hosts.yaml
```

Esse arquivo de inventário é configurado a partir do script [inventory.py](https://github.com/kubernetes-sigs/kubespray/blob/f007c776410433f750bfd62d52e6d10ca5fbd1b8/contrib/inventory_builder/inventory.py#L453). Nesse caso, note que duas variáveis são importantes para isso: `IPS` e `KUBE_CONTROL_HOSTS`.

```bash
CONFIG_FILE=kubespray/inventory/mycluster/hosts.yaml python3 kubespray/contrib/inventory_builder/inventory.py ${IPS[@]}
```

> Pode-se apenas definir a variável `KUBE_CONTROL_HOSTS` nas variáveis de ambiente do repositório e ela será passada diretamente. Essa variável controla quantos nodes devem ser `control planes`. Neste caso, temos definido como `3`, então os nodes 1, 2 e 3 serão configurados como `control planes` no inventário.

Em seguida, são copiados todos os arquivos presentes no repositório para configuração inicial do cluster.

- Para instalar Helm e Krew:
```bash
cp cluster/addons.yml kubespray/inventory/mycluster/group_vars/k8s_cluster/addons.yml
```

- Necessário para configuração gerais do cluster k8s.
```bash
cp cluster/k8s-cluster.yml kubespray/inventory/mycluster/group_vars/k8s_cluster/k8s-cluster.yml
```

- Necessário para configurar a CNI Calino:
```bash
cp cluster/k8s-cluster.yml kubespray/inventory/mycluster/group_vars/k8s_cluster/k8s-net-calico.yml
```

- Necessário para a configuração do coreDNS:
```bash
cp cluster/all.yml kubespray/inventory/mycluster/group_vars/all/all.yml
```

Copiados todos esses arquivos, é necessário realizar a configuração dos arquivos SSH para acesso remotos das VMs. Para tanto, é necessário que os comandos a seguir sejam executados:

```bash
export ANSIBLE_HOST_KEY_CHECKING=False
```

```bash
eval $(ssh-agent -s)
```

```bash
mkdir ~/.ssh
```

```bash
bash -c 'ssh-add <(echo "$SSH_PRIVATE_KEY")'
```

```bash
ssh-keyscan -t rsa $HOSTS >> ~/.ssh/known_hosts
```

Configurado essas informações intermediárias, basta executar o comando a seguir para iniciar o processo deploy do cluster:

```bash
cd kubespray && ansible-playbook -i inventory/mycluster/hosts.yaml cluster.yml -b -v
```

Repare que neste comando final, o arquivo de inventário `hosts.yaml` foi devidamente referenciado. Explicando, portanto, sua importância.

Ao fim dessa explicação, é necessário ressaltar o seguinte ponto: todos esses detalhes acima, que explicitam o job `cluster` dentro do stage `deploy_cluster` partem do pressuposto que todos os arquivos utilizados estão devidamente configurados, para então tudo funcionar como esperado. Observe, portanto, que a documentação dessa parte do projeto não segue a todo momento uma linearidade, algo comum em soluções complexas; tanto que os arquivos mostrados anteriormente nos comandos `"cp"` da pipeline serão explicados a fundo a posteriormente e algumas até dentro de outros jobs. Com isso posto, sempre tenha em mente de ler toda essa documentação antes de executar qualquer comando aqui contido. 

#### Configurando o arquivo addons.yml
O Kubespray permite incluir algumas ferramentas durante o deploy do cluster por meio do arquivo `addons.yml`. Porém, optamos por manter apenas o `Helm` e o `Krew` por exigirem menos configurações, enquanto que as outras ferramentas serão instaladas e configuradas externamente por meio de outros jobs.

Para instalar essas ferramentas, basta alterar os seguintes campos para `true`:

```yaml
helm_enabled: true
...
krew_enabled: true
```

#### Configurando o arquivo k8s-cluster.yml
Para modificar mais configurações do cluster, como o runtime, a versão do kubernetes, a CNI etc., deve-se editar o arquivo `k8s-cluster.yml`. Neste caso, as alterações e campos correspondentes foram os seguintes:

- Versão do kubernetes para `1.26.4` por conta da [compatibilidade com o Rancher](https://www.suse.com/suse-rancher/support-matrix/all-supported-versions/rancher-v2-7-5/):
```yaml
kube_version: v1.26.4
```

- CNI para Calico:
```yaml
kube_network_plugin: calico
```

- Habilitar o ARP para o [funcionamento do Metallb](https://github.com/kubernetes-sigs/kubespray/blob/master/docs/metallb.md#prerequisites)
```yaml
kube_proxy_strict_arp: true
```

- Nome do cluster:
```yaml
cluster_name: cluster.local
```

- Configurar zona para o Node Local DNS
```yaml
nodelocaldns_external_zones:
 - zones:
   - ufpa.br
   nameservers:
   - 8.8.8.8
   - 8.8.4.4
   cache: 5
```

- Habilitar patches do kubeadm:
```yaml
kubeadm_patches:
  enabled: true
```

#### Configurando o arquivo k8s-net-calico.yml
Pode-se alterar configurações mais específicas do Calino no arquivo k8s-net-calico.yml. Neste caso, foi feita apenas a alteração do datastore que o Calico utilizará:
```yaml
calico_datastore: "etcd"
```

#### Configurando o CoreDNS - all.yml
Para a configuração do CoreDNS ocorrer sem problemas é necessário configurar (descomentar) no arquivo `all.yml` alguns resolvedores de nomes. Tais nameservers são os clássicos já conhecidos: `8.8.4.4` e `8.8.8.8`, seguindo o padrão abaixo:

```yaml
upstream_dns_servers:
   - 8.8.8.8
   - 8.8.4.4
```
### Deploy apps

Como comentado anteriormente, a instalação de ferramentas adicionais será feita externamente por outros jobs ao invés do arquivo `addons.yml` disponibilizado pelo Kubespray para que a manutenção seja feita mais facilmente com Helm. Esse stage contém esses jobs e as etapas necessárias para a instalação e configuração dessas ferramentas.

OBS: para que o gitlab tenha acesso ao cluster e possa rodar os comandos indicados nos jobs abaixo, é preciso incluir o arquivo kubeconfig nas variáveis de ambiente com o tipo arquivo (file). Para isso, basta acessar uma vm que serve como node master e copiar o conteúdo do arquivo `.kube/config`.

#### Pre Config

Esse job prepara o ambiente e aplica algumas CRDs necessárias para o funcionamento de algumas ferramentas.

Inicialmente, são aplicados os CRDs para o cert-manager e para o ingress:
```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.crds.yaml
helm pull oci://ghcr.io/nginxinc/charts/nginx-ingress --untar --version 0.18.1
cd nginx-ingress && kubectl apply -f crds/
```

Por fim, adicionamos alguns recursos para o longhorn:
- Primeiro instalamos os pacotes `ICSI` e `NFS` por meio de manifestos para que o longhorn [funcione corretamente](https://longhorn.io/docs/1.5.1/deploy/install/#installation-requirements)
```bash
kubectl apply -f https://raw.githubusercontent.com/longhorn/longhorn/v1.5.1/deploy/prerequisite/longhorn-iscsi-installation.yaml
kubectl apply -f https://raw.githubusercontent.com/longhorn/longhorn/v1.5.1/deploy/prerequisite/longhorn-nfs-installation.yaml
```
- Em seguida, é feito o deploy de um backupstore por meio do arquivo `addons/longhorn/backupstore.yml`, o qual é baseado [neste](https://github.com/longhorn/longhorn/blob/v1.5.1/deploy/backupstores/nfs-backupstore.yaml)
```bash
kubectl apply -f addons/longhorn/backupstore.yml
```

#### MetalLB
Em cluster Bare metal, ou seja, cluster k8s que não estão sob a infraestrutura de nuvem pública como Google Cloud, Azure, AWS e assim por diante; é necessário implantar e configurar um balanceador de carga que irá cuidar de alocar os devidos IPs aos serviços do tipo LoadBalancer que são utilizados para expor seus Pods ao mundo externo. Assim, uma solução para isso é uso do MetalLB. As configurações feitas na pipeline seguem a seguinte ordem:

- Criação do namespace do MetalLB e adição de tags exigidas após a versão 1.21 do Kubernetes

```bash
kubectl create namespace metallb-system
kubectl label namespaces metallb-system pod-security.kubernetes.io/enforce=privileged pod-security.kubernetes.io/audit=privileged pod-security.kubernetes.io/warn=privileged --overwrite=true
```  
- Instalação via helm

```bash
helm repo add metallb https://metallb.github.io/metallb
helm upgrade metallb metallb/metallb -i -n metallb-system
```

Após esses passos o MetalLB estará instalado porém em modo IDLE até que sejam aplicadas as suas configurações. Configurações estas que estão separadas em dois arquivos: `ip_pool.yml` e `l2ad.yml` e serão aplicadas no job `Post Config`

- ip_pool.yml:
```yaml
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: public-ips
  namespace: metallb-system
spec:
  addresses:
  - {initIP}-{endIP}

```

Como o nome sugere, aqui é definido os IPs que o MetalLB poderá usar, além de um nome que será usado posteriormente para identificar a pool ou range de IPs.

- l2ad.yml:
```yaml
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: l2ad
  namespace: metallb-system
spec:
  ipAddressPools:
  - public-ips
```
Com os IPs conhecidos pelo MetalLB no passo anterior, agora definimos qual protocolo será usado para comunicação, nesse caso o Layer 2, note que o nome definido no arquivo anterior é usado aqui.

É necessário, também, que a flag para o protocolo ARP esteja ativa no cluster, para que o MetalLB possa funcionar sem problemas. Essa flag está no arquivo `k8s-cluster.yml`

```yaml
kube_proxy_strict_arp: true
```
MetalLB pode gerenciar várias pools de IPs se for necessário, se esse for o caso é preciso definir uma das pools com `autoAssign: true` nas spec do yaml.

#### Ingress
O job para instalação do Ingress Nginx é bem simples e consiste apenas no seguinte comando:
```bash
helm upgrade nginx-ingress oci://ghcr.io/nginxinc/charts/nginx-ingress -i -f addons/nginx-ingress/values.yml -n nginx-ingress --create-namespace --version 0.18.1
```

Porém, deve-se atentar ao arquivo de values utilizado em `addons/nginx-ingress/values.yml` e modificar de acordo com o contexto. Neste caso, foram feitas as seguintes alterações:

- Deploy do ingress foi como DaemonSet como recomendado pelo [Rancher](https://ranchermanager.docs.rancher.com/v2.6/pages-for-subheaders/installation-requirements#ingress):
```yaml
kind: daemonset
```

- Ingress class `nginx` foi definido como padrão:
```yaml
setAsDefaultIngress: true
```

- Habilitar TLS Passthrough para o [Argo CD](https://argo-cd.readthedocs.io/en/stable/operator-manual/ingress/#option-1-ssl-passthrough)
```yaml
enableTLSPassthrough: true
```


- Definição do IP do LoadBalancer:
```yaml
loadBalancerIP: "loadBIP"
```

#### Cert Manager
Assim como o job do Ingress, o do Cert Manager é simples e consiste no comando:
```bash
helm upgrade cert-manager jetstack/cert-manager -i -n cert-manager --create-namespace --version v1.12.0
```

Porém, o cert manager precisa de um issuer para criar os certificados no cluster. Então, esse issuer foi configurado no arquivo `addons/cert-manager/issuer.yml` e será aplicado no job `Post Config`

#### Longhorn
Como alguns recursos já foram adicionado para o longhorn, o seu job apenas o instala:
```bash
helm upgrade longhorn longhorn/longhorn -i -n longhorn-system --create-namespace --version 1.5.1
```

#### Rancher
A instalação do Rancher é realizada a partir do comando:

```bash
helm upgrade rancher rancher-stable/rancher -i -n cattle-system --create-namespace --set hostname=example.com --set bootstrapPassword={changepassword}
```

Em que são definidos apenas dois campos:
- `hostname`: endereço que o rancher utilizará para ser acessado
- `bootstrapPassword`: senha inicial para autorizar a redefinição da senha que será usada para acessar o rancher

#### Argo CD
O Argo CD é instalado de modo padrão:
```bash
helm upgrade argocd argo/argo-cd -i -n argo-cd --create-namespace
```

#### Post Config
Este job realiza as configurações finais para o funcionamento ou acesso de alguns serviços:

- Metallb
```bash
kubectl apply -f addons/metallb/ip_pool.yml -n metallb-system
kubectl apply -f addons/metallb/l2ad.yml -n metallb-system
```

- Cert Manager
```bash
kubectl apply -f addons/cert-manager/issuer.yml -n cert-manager
```

- Ingress para o Rancher e para o Argo CD:
```bash
kubectl apply -f addons/rancher/ingress.yml -n cattle-system
kubectl apply -f addons/argo-cd/ingress.yml -n argo-cd
```

# Troubleshootings
- CoreDNS:
  - Um problema que pode vir a acometer o deploy dos Pods associados ao coreDNS é a ausência de DNS nameservers no arquivo all.yml. Esses nameservers são importantes para inicializar tais pods de modo a criar corretamente o arquivo Corefile utilizados por pods correlatos ao coreDNS, como o localDNS, essencial para comunicação entre Pods. Certifique-se de adicioná-los conforme a configuração mostrada.
  
- MetalLB:
  - A ausência do metalLB em clusters k8s bare-metal ocasiona, em serviços do tipo loadbalancers, a espera infinita por algum IP. Logo, se o metalLb não for configurado para realizar essa gerência, nenhum pod poderá vir a ser exposto à internet.

- Versão do K8s:
  - É necessário definir uma versão coerente do k8s a fim de que o serviço do Rancher seja suportado. Se a versão do cluster k8s for maior que `v1.26.4`, o serviço do Rancher não poderá ser instalado. Naturalmente, essa problemática tende a ser temporária, haja vista que o projeto do Rancher tenderá avançar a fim de prover suporte para tais versões.