import re

with open('e:/aitrade/frontend/src/pages/AdminDashboard.tsx', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Add state variables
state_vars = """  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');"""
code = re.sub(r"  const \[username, setUsername\] = useState\(''\);\n  const \[password, setPassword\] = useState\(''\);", state_vars, code)

# 2. handleOpenCreateModal
create_modal = """  const handleOpenCreateModal = (forceRole?: string) => {
    setModalMode('create');
    setUsername('');
    setPassword('');
    setEmail('');
    setPhone('');"""
code = re.sub(r"  const handleOpenCreateModal = \(forceRole\?: string\) => \{\n    setModalMode\('create'\);\n    setUsername\(''\);\n    setPassword\(''\);", create_modal, code)

# 3. handleOpenEditModal
edit_modal = """  const handleOpenEditModal = (account: any) => {
    setModalMode('edit');
    setEditingOldUsername(account.username);
    setUsername(account.username);
    setPassword(account.password || '');
    setEmail(account.email || '');
    setPhone(account.phone || '');"""
code = re.sub(r"  const handleOpenEditModal = \(account: any\) => \{\n    setModalMode\('edit'\);\n    setEditingOldUsername\(account\.username\);\n    setUsername\(account\.username\);\n    setPassword\(account\.password \|\| ''\);", edit_modal, code)

# 4. handleSubmit
submit_push = "storedData.push({ username, password, email, phone, role, permissions });"
code = re.sub(r"storedData\.push\(\{ username, password, role, permissions \}\);", submit_push, code)

submit_map = "u.username === editingOldUsername ? { ...u, username, password, email, phone, role, permissions } : u"
code = re.sub(r"u\.username === editingOldUsername \? \{ \.\.\.u, username, password, role, permissions \} : u", submit_map, code)

# 5. JSX Inputs
old_inputs = '''                </div>
                <div>
                  <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Role</label>'''

new_inputs = """                </div>
                <div>
                  <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Email</label>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors"
                    placeholder="Enter email id"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Phone Number</label>
                  <input
                    type="text"
                    value={phone}
                    onChange={(e) => setPhone(e.target.value.replace(/\\D/g, '').slice(0, 10))}
                    pattern="[0-9]{10}"
                    maxLength={10}
                    className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors"
                    placeholder="Enter 10-digit phone number"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Role</label>"""

code = code.replace(old_inputs, new_inputs)

with open('e:/aitrade/frontend/src/pages/AdminDashboard.tsx', 'w', encoding='utf-8') as f:
    f.write(code)

print("done")
