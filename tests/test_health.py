from tests.conftest import client


def test_root_serves_gradio_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower()


def test_health_returns_expected_shape(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert "n_features" in body
    assert "first_3" in body
    assert isinstance(body["first_3"], list)