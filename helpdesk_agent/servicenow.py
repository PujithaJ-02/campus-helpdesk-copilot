"""Thin ServiceNow Table API client (basic auth, works with a free developer instance)."""
import os

import requests

INCIDENT_FIELDS = (
    "sys_id,number,short_description,description,category,urgency,impact,"
    "priority,state,assignment_group,sys_created_on"
)


class ServiceNowError(RuntimeError):
    pass


class ServiceNowClient:
    def __init__(self, instance=None, user=None, password=None, timeout=15):
        instance = instance or os.environ["SN_INSTANCE"]
        if not instance.startswith("http"):
            instance = f"https://{instance}.service-now.com"
        self.base = instance.rstrip("/") + "/api/now/table"
        self.auth = (user or os.environ["SN_USER"], password or os.environ["SN_PASSWORD"])
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})

    def _request(self, method, table, path="", **kwargs):
        url = f"{self.base}/{table}{path}"
        resp = self.session.request(method, url, auth=self.auth, timeout=self.timeout, **kwargs)
        if resp.status_code >= 400:
            raise ServiceNowError(f"{method} {table} failed: HTTP {resp.status_code} {resp.text[:200]}")
        return resp.json().get("result")

    def list_incidents(self, query="active=true^ORDERBYDESCsys_created_on", limit=10):
        params = {
            "sysparm_query": query,
            "sysparm_limit": limit,
            "sysparm_fields": INCIDENT_FIELDS,
            "sysparm_display_value": "true",
            "sysparm_exclude_reference_link": "true",
        }
        return self._request("GET", "incident", params=params) or []

    def get_incident(self, number):
        rows = self.list_incidents(query=f"number={number}", limit=1)
        return rows[0] if rows else None

    def update_incident(self, sys_id, fields):
        params = {"sysparm_fields": INCIDENT_FIELDS, "sysparm_display_value": "true"}
        return self._request("PATCH", "incident", f"/{sys_id}", params=params, json=fields)

    def create_incident(self, fields):
        params = {"sysparm_fields": INCIDENT_FIELDS, "sysparm_display_value": "true"}
        return self._request("POST", "incident", params=params, json=fields)

    def find_group_id(self, name):
        params = {"sysparm_query": f"name={name}^active=true", "sysparm_limit": 1, "sysparm_fields": "sys_id"}
        rows = self._request("GET", "sys_user_group", params=params) or []
        return rows[0]["sys_id"] if rows else None

    def search_knowledge(self, text, limit=3):
        params = {
            "sysparm_query": f"workflow_state=published^short_descriptionLIKE{text}^ORtextLIKE{text}",
            "sysparm_limit": limit,
            "sysparm_fields": "number,short_description,text",
        }
        return self._request("GET", "kb_knowledge", params=params) or []
