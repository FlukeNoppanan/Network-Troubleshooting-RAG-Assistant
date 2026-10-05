# NetAssist RAG — Knowledge expansion and UI validation

## Scope

Expanded the existing application in place. Preserved all 10 original knowledge documents and added 8 original Thai educational documents (11–18): Ethernet switching, TCP/UDP/ICMP, ARP/MAC tables, wireless, IPv6, network security, services and packet analysis. Added symptoms, possible causes, verification commands and remediation practices. No dependency, second model, database or LangChain was added.

Kept the original 18 regression questions and added 9 questions (27 total), rather than deleting previously validated cases. Four NOT_FOUND cases cover Kubernetes in English and Thai, PostgreSQL and Mobile Legends.

## Validated corpus

- documents: 18
- raw_characters: 38996
- cleaned_characters: 38978
- chunks: 105
- vectors: 105
- dimensions: 384
- max_chunk_tokens: 128
- model_token_limit: 128
- query_norm: 1.0000001192092896
- document_norm_min: 0.9999999403953552
- document_norm_max: 1.0000001192092896
- min_grounded_top_score: 0.4659
- max_not_found_top_score: 0.2883
- threshold: 0.38

All chunks fit the model's 128-token budget. FAISS IndexFlatIP has exactly 105 normalized vectors for 105 chunks. Document/query norms are approximately 1. The minimum grounded top score is 0.4659; the maximum NOT_FOUND top score is 0.2883, so the existing threshold 0.38 remains justified without lowering it.

## Retrieval results

PASS means the expected source is present in the accepted Top-5 for GROUNDED cases, or all results are below threshold for NOT_FOUND. Some expected documents rank 2–3 because new documents contain overlapping, relevant troubleshooting content. The table includes expected rank to make this explicit. Every NOT_FOUND case is also tested to return the exact refusal text without calling Gemini.

| ID | Question | Expected | Expected source | Top source | Cosine | Expected rank | Result |
|---|---|---|---|---|---:|---:|---|
| 1 | OSPF neighbor stuck in EXSTART or EXCHANGE what should I check? | GROUNDED | 05_ospf.txt | 05_ospf.txt | 0.5787 | 1.0 | PASS |
| 2 | เครื่องต่าง VLAN ติดต่อกันไม่ได้ต้องตรวจอะไร? | GROUNDED | 03_vlan_trunking.txt | 03_vlan_trunking.txt | 0.7556 | 1.0 | PASS |
| 3 | DHCP client cannot obtain an IP address how do I troubleshoot? | GROUNDED | 06_dhcp_dns.txt | 06_dhcp_dns.txt | 0.8108 | 1.0 | PASS |
| 4 | DNS resolution fails but access by IP works what should I check? | GROUNDED | 06_dhcp_dns.txt | 06_dhcp_dns.txt | 0.7763 | 1.0 | PASS |
| 5 | How do I troubleshoot outbound NAT translations? | GROUNDED | 07_nat.txt | 07_nat.txt | 0.6436 | 1.0 | PASS |
| 6 | เกิด STP loop และ MAC flapping ต้องตรวจอะไร? | GROUNDED | 04_stp_rstp.txt | 13_arp_mac_table.txt | 0.7463 | 3.0 | PASS |
| 7 | VPN ส่ง packet เล็กได้แต่เว็บค้าง ต้องตรวจ MTU และ MSS อย่างไร? | GROUNDED | 09_vpn.txt | 09_vpn.txt | 0.7933 | 1.0 | PASS |
| 8 | How do I troubleshoot a firewall rule blocking TCP traffic? | GROUNDED | 08_firewall_acl.txt | 08_firewall_acl.txt | 0.7785 | 1.0 | PASS |
| 9 | ตั้ง subnet mask หรือ default gateway ผิดมีอาการอย่างไร? | GROUNDED | 02_ip_subnetting.txt | 03_vlan_trunking.txt | 0.669 | 2.0 | PASS |
| 10 | How do I troubleshoot TCP retransmission and duplicate ACK? | GROUNDED | 10_network_troubleshooting.txt | 10_network_troubleshooting.txt | 0.8072 | 1.0 | PASS |
| 11 | How do I deploy Kubernetes with Helm? | NOT_FOUND | — | 04_stp_rstp.txt | 0.2883 | — | PASS |
| 12 | How do I optimize a PostgreSQL query? | NOT_FOUND | — | 10_network_troubleshooting.txt | 0.2802 | — | PASS |
| 13 | ARP ใช้ทำอะไรเมื่อปลายทางอยู่นอก subnet? | GROUNDED | 01_osi_tcpip.txt | 01_osi_tcpip.txt | 0.7205 | 1.0 | PASS |
| 14 | OSPF DROTHER neighbors stay in 2-Way is that normal? | GROUNDED | 05_ospf.txt | 05_ospf.txt | 0.4659 | 1.0 | PASS |
| 15 | OSPF Neighbor ค้างอยู่ที่ EXSTART เกิดจากอะไร? | GROUNDED | 05_ospf.txt | 05_ospf.txt | 0.6192 | 1.0 | PASS |
| 16 | Client ไม่ได้รับ IP Address จาก DHCP ควรตรวจสอบอะไร? | GROUNDED | 06_dhcp_dns.txt | 06_dhcp_dns.txt | 0.8007 | 1.0 | PASS |
| 17 | VLAN เดียวกันแต่เครื่องสองเครื่องสื่อสารกันไม่ได้ ควรตรวจสอบอะไร? | GROUNDED | 03_vlan_trunking.txt | 03_vlan_trunking.txt | 0.7837 | 1.0 | PASS |
| 18 | วิธี Deploy Kubernetes ด้วย Helm ทำยังไง? | NOT_FOUND | — | 04_stp_rstp.txt | 0.2812 | — | PASS |
| 19 | Wi-Fi ช้าและหลุดบ่อย ควรตรวจ RSSI SNR และ interference อย่างไร? | GROUNDED | 14_wireless_network.txt | 14_wireless_network.txt | 0.7758 | 1.0 | PASS |
| 20 | IPv6 Neighbor Discovery ไม่สำเร็จและไม่มี default route ควรตรวจอะไร? | GROUNDED | 15_ipv6.txt | 15_ipv6.txt | 0.8329 | 1.0 | PASS |
| 21 | MAC address table เรียน MAC สลับ port และ ARP entry เปลี่ยน ควรตรวจอะไร? | GROUNDED | 13_arp_mac_table.txt | 13_arp_mac_table.txt | 0.8216 | 1.0 | PASS |
| 22 | How do I troubleshoot Ethernet speed duplex mismatch and CRC errors? | GROUNDED | 11_ethernet_switching.txt | 11_ethernet_switching.txt | 0.6783 | 1.0 | PASS |
| 23 | Wireshark แสดง TCP Out-of-Order และ checksum incorrect เป็น packet loss จริงไหม? | GROUNDED | 18_packet_analysis.txt | 10_network_troubleshooting.txt | 0.7556 | 3.0 | PASS |
| 24 | NTP ไม่ sync และ SSH connection refused ต้องตรวจ service อะไร? | GROUNDED | 17_network_services.txt | 17_network_services.txt | 0.78 | 1.0 | PASS |
| 25 | 802.1X ผ่านแต่ Dynamic ARP Inspection drop ARP ควรตรวจอะไร? | GROUNDED | 16_network_security.txt | 13_arp_mac_table.txt | 0.6401 | 2.0 | PASS |
| 26 | Mobile Legends ควรเลือกฮีโร่ตัวไหนและออกไอเทมอย่างไร? | NOT_FOUND | — | 14_wireless_network.txt | 0.1841 | — | PASS |
| 27 | Destination host unreachable ต่างจาก Request timed out อย่างไร และควรตรวจ ICMP ผู้ส่งอย่างไร? | GROUNDED | 12_tcp_udp_icmp.txt | 12_tcp_udp_icmp.txt | 0.7995 | 1.0 | PASS |

## UI and integration

NetAssist RAG identity, restrained header and capability tags, responsive native centered layout, readiness status, sidebar statistics/coverage, compact system information and Advanced RAG Details. The welcome panel has four native buttons that submit through the same RAG pipeline. Clear Chat restores the welcome state. Sources show filename, chunk, cosine and a preview; full evidence remains available in an expander. Developer debug remains opt-in. Refusal note is a separate caption and does not change the answer string. CSS targets only app-owned classes and does not hide Streamlit controls.

AST comparison confirms generate_answer, retrieve_chunks, build_faiss_index, load_embedding_model and validate_answer are unchanged from the published version. Gemini remains gemini-3.1-flash-lite. Automated integration uses a mocked official client; it checks the grounded OSPF chat, citations, source expander and debug. No additional paid/live Gemini call was made in this UI expansion pass. Earlier live integration evidence is historical, not a new test result.

## Verification

- `HF_HUB_OFFLINE=1 .venv/bin/python test_rag.py`: 8 tests passed, including all 27 retrieval cases, normalization/source mapping, token limits, citation/generation regression, missing secrets, session history, Clear Chat and example-button submission.
- `HF_HUB_OFFLINE=1 .venv/bin/python debug_retrieval.py`: all 27 cases passed; results saved in CSV and JSON.
- `pip check` and Python compile: passed.
- Local Streamlit started at http://127.0.0.1:8504; health endpoint returned `ok`.
- Source-file scan found no likely credential patterns. Real secrets were neither read for scanning nor modified; .streamlit/secrets.toml remains ignored.
- No commit, push or cloud deployment was performed.
- No browser visual inspection; UI behavior verified using Streamlit AppTest. Mobile styling uses native layout, a wrapping chip row and a small app-owned media rule; no real-device screenshot validation.

## Files

Added: data/11_ethernet_switching.txt through data/18_packet_analysis.txt and this report.
Modified: app.py, .streamlit/config.toml, test_questions.csv, test_rag.py, README.md, diagnostics/retrieval_results.csv, diagnostics/retrieval_details.json.

## Run locally

```bash
.venv/bin/streamlit run app.py
```

Cloud will only receive these changes after a future authorized commit/push. The existing deployment is unchanged in this task.
