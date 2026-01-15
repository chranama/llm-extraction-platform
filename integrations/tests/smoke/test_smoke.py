def test_readyz(client):
    r = client.get("/readyz")
    assert r.status_code == 200

def test_modelz(client):
    r = client.get("/modelz")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)