import unittest

from lan_access import discover_lan_access


class LanAccessTests(unittest.TestCase):
    def test_filters_and_prefers_private_route(self):
        info = discover_lan_access(
            8123,
            candidate_provider=lambda: [
                "127.0.0.1",
                "169.254.2.3",
                "8.8.8.8",
                "192.168.1.44",
                "10.0.0.7",
                "192.168.1.44",
            ],
            preferred_provider=lambda: "10.0.0.7",
        )

        self.assertEqual("http://10.0.0.7:8123", info.preferred_url)
        self.assertEqual(
            ("http://10.0.0.7:8123", "http://192.168.1.44:8123"),
            info.phone_urls,
        )

    def test_no_lan_keeps_loopback(self):
        info = discover_lan_access(
            7860,
            candidate_provider=lambda: ["127.0.0.1"],
            preferred_provider=lambda: None,
        )

        self.assertEqual("http://127.0.0.1:7860", info.loopback_url)
        self.assertIsNone(info.preferred_url)
        self.assertEqual((), info.phone_urls)

    def test_provider_failures_fall_back_to_loopback(self):
        def fail():
            raise OSError("adapter lookup failed")

        info = discover_lan_access(
            9000,
            candidate_provider=fail,
            preferred_provider=fail,
        )

        self.assertEqual("http://127.0.0.1:9000", info.loopback_url)
        self.assertEqual((), info.phone_urls)

    def test_rejects_malformed_public_and_non_rfc1918_addresses(self):
        info = discover_lan_access(
            7860,
            candidate_provider=lambda: [
                "not-an-address",
                "0.0.0.0",
                "224.0.0.1",
                "100.64.0.1",
                "172.15.0.1",
                "172.16.0.1",
                "172.31.255.254",
                "172.32.0.1",
            ],
            preferred_provider=lambda: "8.8.4.4",
        )

        self.assertEqual(
            (
                "http://172.16.0.1:7860",
                "http://172.31.255.254:7860",
            ),
            info.phone_urls,
        )


if __name__ == "__main__":
    unittest.main()
