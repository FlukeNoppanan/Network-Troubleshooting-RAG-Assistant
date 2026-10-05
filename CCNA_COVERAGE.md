# คลังความรู้พื้นฐาน CCNA ของ NetAssist RAG

ขอบเขตหัวข้ออ้างอิง [Cisco CCNA 200-301 v1.1 Exam Topics](https://learningcontent.cisco.com/documents/marketing/exam-topics/200-301-CCNA-v1.1.pdf) เพื่อจัดหมวดเท่านั้น เนื้อหาใน `data/` เป็นคำอธิบายที่เขียนใหม่ ไม่ได้นำ PDF หรือข้อสอบ Cisco เข้า index ไม่ใช่ข้อสอบจริงหรือเอกสารรับรองจาก Cisco

| หมวด | สิ่งที่อ่านและฝึกได้ | เอกสารหลักใน data/ |
|---|---|---|
| Network Fundamentals | OSI/TCP-IP, components/topologies, media/PoE, Ethernet, IPv4/CIDR/VLSM, IPv6, virtualization/VRF | 01, 02, 11–15, 19–22, 25, 44 |
| Network Access | VLAN/trunk, inter-VLAN, STP/protection, LACP, CDP/LLDP, WLAN/WLC และ wireless security | 03, 04, 14, 28–33 |
| IP Connectivity | routing table/LPM, static/default/floating routes, IPv6 routes, single-area OSPF และ FHRP | 05, 23–27 |
| IP Services | DHCP/DNS, NAT/PAT, NTP/SNMP/syslog, SSH, QoS และ file-transfer services | 06, 07, 17, 35, 37–40 |
| Security Fundamentals | firewall/ACL, VPN, policy/identity/AAA, wireless security และ Layer 2 protections | 08, 09, 16, 33, 36, 41–43 |
| Automation and Programmability | SDN/controllers, underlay/overlay, REST/CRUD/JSON, Ansible/Terraform และ AI/ML ในงานเครือข่าย | 44–48 |
| การใช้หลักฐานแก้ incident | packet capture, retransmission, PMTUD, ARP/MAC, loss, latency และการ verify/rollback | 10–18, 20, 23, 34 |

แต่ละไฟล์มีพื้นฐานตามหัวข้อ อาการที่พบบ่อย วิธีตรวจ และข้อควรระวัง ใช้ภาษาไทยพร้อมศัพท์เทคนิคอังกฤษ คำสั่ง Cisco IOS เป็นตัวอย่างสำหรับ lab และต้องตรวจชื่อ interface, IP และ feature ของรุ่นจริงก่อนใช้

## รายชื่อเอกสารที่เพิ่มจากชุด 18 ไฟล์

| ไฟล์ | หัวข้อ |
|---|---|
| 19_network_architectures.txt | Components, campus, WAN, SOHO, cloud และ spine-leaf |
| 20_cabling_transceivers_poe.txt | Copper/fiber, optics และ PoE |
| 21_subnetting_binary_cidr.txt | Binary, CIDR, จำนวน hosts และ private addressing |
| 22_vlsm_summarization.txt | VLSM และ route summaries พร้อมตัวอย่างคำนวณ |
| 23_routing_table_forwarding.txt | LPM, administrative distance, metric และ return paths |
| 24_static_default_floating_routes.txt | IPv4 static/default/host/floating routes |
| 25_ipv6_addresses_static_routes.txt | IPv6 types, EUI-64 และ static routes |
| 26_single_area_ospfv2_lab.txt | OSPFv2 lab, wildcard, passive interfaces และ verification |
| 27_first_hop_redundancy.txt | Virtual gateway, priorities และ failover |
| 28_etherchannel_lacp.txt | LACP negotiation, hashing และ bundled ports |
| 29_cdp_lldp_discovery.txt | Neighbor discovery และการยืนยัน cabling |
| 30_intervlan_svi_router_on_stick.txt | SVI และ router-on-a-stick lab |
| 31_spanning_tree_protection.txt | Root Guard, Loop Guard และ BPDU controls |
| 32_wireless_controller_architecture.txt | AP modes, CAPWAP, WLC และ WLAN mapping |
| 33_wireless_security_wpa.txt | WPA2/WPA3, PSK, Enterprise และ certificates |
| 34_cisco_ios_basics.txt | CLI modes, running/startup config และ rollback |
| 35_ssh_device_management.txt | SSH/VTY, console, management ACL และ VRF |
| 36_standard_extended_acls_lab.txt | Standard/extended ACL, wildcard และ direction |
| 37_nat_inside_source_lab.txt | Static NAT, pools และ overload lab |
| 38_dhcp_client_relay_lab.txt | DHCP client/helper, giaddr และ scope exhaustion |
| 39_ntp_snmp_syslog.txt | Time sync, polling/traps, MIB/OID และ severity |
| 40_qos_classification_queuing.txt | Classification, marking, queues, policing และ shaping |
| 41_aaa_radius_tacacs.txt | AAA, RADIUS, TACACS+ และ fallback |
| 42_layer2_security_lab.txt | Snooping, DAI และ port security |
| 43_security_fundamentals_policy.txt | Threat/risk, identity, segmentation และ incident controls |
| 44_virtualization_vrf_cloud.txt | VMs, containers, VRF และ cloud paths |
| 45_sdn_controller_apis.txt | Control/data planes, fabric และ controller APIs |
| 46_rest_api_json_fundamentals.txt | REST methods/status, authentication และ JSON |
| 47_ansible_terraform_config_management.txt | Configuration management, state, plans และ drift |
| 48_ai_ml_network_operations.txt | Predictive/generative AI, telemetry และ cloud management |

## แนวทางฝึกใช้งาน

1. อ่าน OSI และ addressing แล้วลองคำนวณ subnet/VLSM ก่อนดูตัวอย่างคำตอบ
2. ฝึก switching: access/trunk → inter-VLAN → STP → EtherChannel พร้อมตรวจ topology
3. ฝึก routing: routing table → static/floating routes → OSPF → FHRP
4. ตรวจ services, management และ security พร้อมทดสอบ traffic ที่อนุญาตและถูกปฏิเสธ
5. อ่าน REST/JSON/controller concepts แล้วประเมินผลกระทบก่อนฝึก automation ใน lab
6. สำหรับทุก incident ให้เก็บหลักฐาน ทดสอบสมมติฐาน เปลี่ยนทีละอย่าง แล้ว verify/rollback

ชุดนี้ครอบคลุมหมวดพื้นฐาน แต่ไม่แทนการลง lab, การอ่านคู่มือ platform หรือการเตรียมสอบแบบเต็มหลักสูตร ไม่มีการรับประกันคะแนนสอบ บางคำถามที่ต้องการรายละเอียดเฉพาะ vendor/version อาจยังไม่มีหลักฐานพอและระบบจะปฏิเสธตามเดิม
