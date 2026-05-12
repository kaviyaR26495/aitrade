import re

with open('e:/aitrade/frontend/src/pages/AdminDashboard.tsx', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Add currentUser
old_role_decl = "const currentUserRole = localStorage.getItem('aitrade-current-role');"
new_role_decl = "const currentUserRole = localStorage.getItem('aitrade-current-role');\n  const currentUser = localStorage.getItem('aitrade-current-user');"
if "const currentUser =" not in code:
    code = code.replace(old_role_decl, new_role_decl)

# 2. Add createdBy to handleSubmit
old_push = "storedData.push({ username, password, email, phone, role, permissions });"
new_push = "storedData.push({ username, password, email, phone, role, permissions, createdBy: currentUser });"
code = code.replace(old_push, new_push)

# 3. Update table filter
old_filter = "accounts.filter(a => currentUserRole === 'admin' ? (a.role !== 'super_admin' && a.role !== 'admin') : true)"
new_filter = "accounts.filter(a => currentUserRole === 'admin' ? (a.role !== 'super_admin' && a.role !== 'admin' && a.createdBy === currentUser) : true)"
code = code.replace(old_filter, new_filter)

with open('e:/aitrade/frontend/src/pages/AdminDashboard.tsx', 'w', encoding='utf-8') as f:
    f.write(code)

print("done")
